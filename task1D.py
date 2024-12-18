import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization."""
        self.steps.append((description, image.copy()))

    # -------------------------
    # Utility Functions
    # -------------------------
    def aggressive_contrast_stretch(self, image):
        """Apply aggressive contrast stretching for extremely low contrast images."""
        min_val, max_val = np.min(image), np.max(image)
        if max_val == min_val:
            return image
        stretched = ((image - min_val)/(max_val - min_val)*255).astype(np.uint8)
        return stretched

    def estimate_bar_thickness(self, image):
        """Estimate the typical bar thickness using vertical runs of black pixels after adaptive binarization."""
        temp_binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 10
        )
        heights = []
        for col in range(temp_binary.shape[1]):
            column = temp_binary[:, col]
            run_length = 0
            black_runs = []
            for val in column:
                if val == 0:
                    run_length += 1
                else:
                    if run_length > 0:
                        black_runs.append(run_length)
                    run_length = 0
            if run_length > 0:
                black_runs.append(run_length)
            if black_runs:
                heights.extend(black_runs)
        if not heights:
            return 5  # Default fallback if no bars are detected
        return int(np.median(heights))

    def get_anomalous_regions(self, anomalies, bar_thickness):
        """Group contiguous anomalous indices into regions, considering bar_thickness."""
        regions = []
        if len(anomalies) == 0:
            return regions
        current_region = [anomalies[0]]
        for i in range(1, len(anomalies)):
            if anomalies[i] == anomalies[i-1] + 1 and (anomalies[i] - current_region[0]) < bar_thickness*2:
                current_region.append(anomalies[i])
            else:
                regions.append(current_region)
                current_region = [anomalies[i]]
        regions.append(current_region)
        return regions

    def detect_compression_artifacts(self, gray):
        """Heuristic detection of compression artifacts based on high-frequency noise in the Fourier domain."""
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        rows, cols = magnitude_spectrum.shape
        # Sample corners for high-frequency noise
        corner_std = (np.std(magnitude_spectrum[:rows//4, :cols//4]) +
                      np.std(magnitude_spectrum[:rows//4, 3*cols//4:]) +
                      np.std(magnitude_spectrum[3*rows//4:, :cols//4]) +
                      np.std(magnitude_spectrum[3*rows//4:, 3*cols//4:]))
        return corner_std > 15

    def detect_image_problems(self, image):
        """Detect image problems: blur, low_contrast, salt_pepper, foreign_object, compression_artifacts, frequency_domain, rotation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

        # Preliminary frequency domain check (for foreign object analysis)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        freq_problem = np.std(magnitude_spectrum) > 45

        # Preliminary contrast check
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()
        cumsum = np.cumsum(hist_norm)
        idx_5 = np.where(cumsum>=0.05)[0][0]
        idx_95 = np.where(cumsum>=0.95)[0][0]
        low_contrast_detected = (idx_95 - idx_5) < 50

        # Temporary processing for bar thickness estimation (if needed)
        temp_processed = gray.copy()
        if freq_problem:
            temp_processed = self.fix_frequency_domain(temp_processed)
        if low_contrast_detected:
            temp_processed = self.fix_contrast(temp_processed)

        bar_thickness = self.estimate_bar_thickness(temp_processed)

        # Profiles for foreign object detection
        vertical_profile = np.mean(gray, axis=0)
        horizontal_profile = np.mean(gray, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        v_anomalies = np.where(np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * v_std)[0]
        h_anomalies = np.where(np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3.0 * h_std)[0]
        v_anomalous_regions = self.get_anomalous_regions(v_anomalies, bar_thickness)
        h_anomalous_regions = self.get_anomalous_regions(h_anomalies, bar_thickness)

        problems = []

        if v_anomalous_regions or h_anomalous_regions:
            problems.append('foreign_object')

        # Blur detection: use Laplacian var + edge density
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        # If Laplacian var < 90 and edge density < 0.15, consider blur
        if laplacian_var < 90 and edge_density < 0.15:
            problems.append('blur')

        # Low contrast
        if low_contrast_detected:
            problems.append('low_contrast')

        # Salt & pepper
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.percentile(noise_diff, 95) > 30:
            problems.append('salt_pepper')

        # Compression artifacts
        if self.detect_compression_artifacts(gray):
            problems.append('compression_artifacts')

        # Frequency domain
        if freq_problem:
            problems.append('frequency_domain')

        # Rotation detection
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        return problems

    def fix_frequency_domain(self, image):
        """Fix frequency domain issues with Gaussian high-pass filter."""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        sigma = 30
        x = np.linspace(-ccol, ccol-1, cols)
        y = np.linspace(-crow, crow-1, rows)
        xv, yv = np.meshgrid(x, y)
        gaussian_filter = np.exp(-(xv**2 + yv**2)/(2*sigma**2))
        mask = 1 - gaussian_filter
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def remove_salt_and_pepper(self, image):
        """Enhanced salt and pepper removal using iterative median filtering."""
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        noise_level = np.std(result - image)
        if noise_level > 20:
            result = cv2.medianBlur(result, 3)
        return result

    def fix_blur(self, image):
        """Two-stage blur correction: unsharp + optional sharpening."""
        gaussian = cv2.GaussianBlur(image, (0,0), 2)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if lap_var < 150:
            kernel = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
            return cv2.filter2D(unsharp, -1, kernel)
        return unsharp

    def fix_contrast(self, image):
        """Fix contrast using CLAHE and mild gamma correction."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        # Mild gamma correction if needed
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)
        return enhanced

    def fix_compression_artifacts(self, image):
        """Mild non-local means denoising to reduce compression artifacts."""
        return cv2.fastNlMeansDenoising(image, None, 5, 7, 21)

    def binarize(self, image):
        """Binarize carefully to avoid holes."""
        p2, p98 = np.percentile(image, (2,98))
        if p2 == p98:
            normalized = image
        else:
            normalized = np.interp(image, (p2,p98), (0,255)).astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            normalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            5
        )
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def remove_foreign_object(self, image):
        """Remove foreign objects using bar thickness and interpolation."""
        gray = image.copy()
        # Re-check problems for frequency domain and contrast:
        # We'll just attempt improvements if needed
        # This is a heuristic; you could store initial problems globally if needed.
        # For simplicity, we rely on previously detected 'problems' global var.
        global problems
        temp_processed = gray.copy()
        if 'frequency_domain' in problems:
            temp_processed = self.fix_frequency_domain(temp_processed)
        if 'low_contrast' in problems:
            temp_processed = self.fix_contrast(temp_processed)
        bar_thickness = self.estimate_bar_thickness(temp_processed)

        vertical_profile = np.mean(gray, axis=0)
        horizontal_profile = np.mean(gray, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0*v_std
        h_anomalous = np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3.0*h_std
        v_anomalous_regions = self.get_anomalous_regions(np.where(v_anomalous)[0], bar_thickness)
        h_anomalous_regions = self.get_anomalous_regions(np.where(h_anomalous)[0], bar_thickness)

        cleaned = gray.copy()
        # Vertical anomalies
        for region in v_anomalous_regions:
            for i in region:
                left_idx = i-1
                while left_idx >= 0 and left_idx >= i-bar_thickness and v_anomalous[left_idx]:
                    left_idx -= 1
                right_idx = i+1
                while right_idx<gray.shape[1] and right_idx <= i+bar_thickness and v_anomalous[right_idx]:
                    right_idx += 1
                if left_idx >= 0 and left_idx >= i-bar_thickness and not v_anomalous[left_idx]:
                    left_values = gray[:, max(0,left_idx-bar_thickness):left_idx+1]
                    cleaned[:, i] = np.mean(left_values, axis=1)
                elif right_idx<gray.shape[1] and right_idx <= i+bar_thickness and not v_anomalous[right_idx]:
                    right_values = gray[:, right_idx:min(gray.shape[1], right_idx+bar_thickness+1)]
                    cleaned[:, i] = np.mean(right_values, axis=1)

        # Horizontal anomalies
        for region in h_anomalous_regions:
            for j in region:
                top_idx = j-1
                while top_idx >= 0 and top_idx >= j-bar_thickness and h_anomalous[top_idx]:
                    top_idx -= 1
                bottom_idx = j+1
                while bottom_idx<cleaned.shape[0] and bottom_idx <= j+bar_thickness and h_anomalous[bottom_idx]:
                    bottom_idx += 1
                if top_idx >=0 and top_idx >= j-bar_thickness and not h_anomalous[top_idx]:
                    top_values = cleaned[max(0, top_idx-bar_thickness):top_idx+1, :]
                    cleaned[j, :] = np.mean(top_values, axis=0)
                elif bottom_idx<cleaned.shape[0] and bottom_idx <= j+bar_thickness and not h_anomalous[bottom_idx]:
                    bottom_values = cleaned[bottom_idx:min(cleaned.shape[0], bottom_idx+bar_thickness+1), :]
                    cleaned[j, :] = np.mean(bottom_values, axis=0)
        return cleaned

    def fix_rotation(self, image):
        """Fix rotation with padding to avoid losing barcode parts."""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = []
            for rho, theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            if angles:
                median_angle = np.median(angles)
                rows, cols = image.shape
                angle_rad = np.radians(abs(median_angle))
                new_width = int(cols*np.abs(np.cos(angle_rad))+rows*np.abs(np.sin(angle_rad)))
                new_height = int(rows*np.abs(np.cos(angle_rad))+cols*np.abs(np.sin(angle_rad)))
                pad_x = (new_width - cols) // 2
                pad_y = (new_height - rows) // 2
                M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                rotated = cv2.warpAffine(image, M, (new_width, new_height),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                # Add padding
                rotated = cv2.copyMakeBorder(rotated, pad_y, pad_y, pad_x, pad_x,
                                             cv2.BORDER_CONSTANT, value=255)
                return rotated
        return image

    def crop_to_barcode(self, binary):
        """Crop to the barcode region based on aspect ratio and area."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = w * h
            if 1.5 < aspect_ratio < 5 and area > (binary.shape[0]*binary.shape[1]*0.05):
                valid_contours.append((contour, area))
        if valid_contours:
            largest_contour = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Add padding
            padding_x = int(0.1*w)
            padding_y = int(0.1*h)
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(binary.shape[1] - x, w + 2*padding_x)
            h = min(binary.shape[0] - y, h + 2*padding_y)
            return binary[y:y+h, x:x+w]
        return binary

    def preprocess_image(self, image):
        """Main preprocessing pipeline."""
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image
        global problems
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        processed = gray.copy()

        # Foreign object first
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed)
            self.add_step("Removed Foreign Object", processed)

        # Frequency domain second
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed)
            self.add_step("Fixed Frequency Domain", processed)

        # Compression artifacts
        if 'compression_artifacts' in problems:
            processed = self.fix_compression_artifacts(processed)
            self.add_step("Fixed Compression Artifacts", processed)

        # Salt & Pepper
        if 'salt_pepper' in problems:
            processed = self.remove_salt_and_pepper(processed)
            self.add_step("Removed Salt & Pepper", processed)

        # Blur
        if 'blur' in problems:
            processed = self.fix_blur(processed)
            self.add_step("Fixed Blur", processed)

        # Low contrast
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed)
            self.add_step("Enhanced Contrast", processed)

        # Binarize
        binary = self.binarize(processed)
        self.add_step("Binarized", binary)

        # Rotation last to avoid losing barcode parts
        if 'rotation' in problems:
            binary = self.fix_rotation(binary)
            self.add_step("Corrected Rotation", binary)

        # Crop
        cropped = self.crop_to_barcode(binary)
        self.add_step("Cropped Barcode", cropped)

        return cropped

    def visualize_steps(self):
        """Visualize all processing steps."""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows*cols == 1:
            axes = np.array([axes])
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        for i, (description, img) in enumerate(self.steps):
            r = i // cols
            c = i % cols
            ax = axes[r, c]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(description, fontsize=8)
            ax.axis('off')
        for i in range(len(self.steps), rows*cols):
            r = i // cols
            c = i % cols
            axes[r, c].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    preprocessor = BarcodePreprocessor()
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()
