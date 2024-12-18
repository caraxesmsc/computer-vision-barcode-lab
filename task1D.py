import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    # -------------------------
    # Problem Detection
    # -------------------------
    def detect_image_problems(self, image):
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

        # --- Foreign object detection (Using codebase 7's improved logic + peak detection) ---
        
        temp_processed = gray.copy()
        
        # Fix frequency domain and contrast issues for bar thickness estimation only if they are detected
        temp_problems = []
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            temp_problems.append('frequency_domain')
            
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()
        cumsum = np.cumsum(hist_norm)
        idx_5 = np.where(cumsum>=0.05)[0][0]
        idx_95 = np.where(cumsum>=0.95)[0][0]
        if (idx_95 - idx_5) < 50:
            temp_problems.append('low_contrast')
        
        if 'frequency_domain' in temp_problems:
            temp_processed = self.fix_frequency_domain(temp_processed)
        if 'low_contrast' in temp_problems:
            temp_processed = self.fix_contrast(temp_processed)
        
        bar_thickness = self.estimate_bar_thickness(temp_processed)
        
        
        vertical_profile = np.mean(gray, axis=0)
        horizontal_profile = np.mean(gray, axis=1)

        
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)

        
        v_anomalies = np.where(np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * v_std)[0]
        h_anomalies = np.where(np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3.0 * h_std)[0]

        
        v_anomalous_regions = self.get_anomalous_regions(v_anomalies, bar_thickness)
        h_anomalous_regions = self.get_anomalous_regions(h_anomalies, bar_thickness)

        
        if len(v_anomalous_regions) > 0 or len(h_anomalous_regions) > 0:
            problems.append('foreign_object')

        # --- Blur detection (Slightly more sensitive threshold) ---
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 90:  # Increased sensitivity (from 80)
            problems.append('blur')

        # --- Low contrast detection (Codebase 1) ---
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()
        cumsum = np.cumsum(hist_norm)
        idx_5 = np.where(cumsum>=0.05)[0][0]
        idx_95 = np.where(cumsum>=0.95)[0][0]
        if (idx_95 - idx_5) < 50:
            problems.append('low_contrast')

        # --- Salt and pepper detection (Codebase 1) ---
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.mean(noise_diff) > 10:
            problems.append('salt_pepper')
            
        # --- Compression artifacts detection (heuristic) ---
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        
        high_freq_noise = np.std(magnitude_spectrum[0:magnitude_spectrum.shape[0]//4, 0:magnitude_spectrum.shape[1]//4]) + \
                          np.std(magnitude_spectrum[0:magnitude_spectrum.shape[0]//4, 3*magnitude_spectrum.shape[1]//4:]) + \
                          np.std(magnitude_spectrum[3*magnitude_spectrum.shape[0]//4:, 0:magnitude_spectrum.shape[1]//4]) + \
                          np.std(magnitude_spectrum[3*magnitude_spectrum.shape[0]//4:, 3*magnitude_spectrum.shape[1]//4:])

        
        if high_freq_noise > 15:
            problems.append('compression_artifacts')

        # --- Frequency domain detection (Codebase 2) ---
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # --- Rotation detection (Codebase 1) ---
        edges = cv2.Canny(gray, 100, 200)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        return problems

    def get_anomalous_regions(self, anomalies, bar_thickness):
        """Helper function to group contiguous anomalous regions"""
        regions = []
        if len(anomalies) > 0:
            current_region = [anomalies[0]]
            for i in range(1, len(anomalies)):
                if anomalies[i] == anomalies[i-1] + 1 and anomalies[i] - current_region[0] < bar_thickness * 2:
                    current_region.append(anomalies[i])
                else:
                    regions.append(current_region)
                    current_region = [anomalies[i]]
            regions.append(current_region)
        return regions
    # -------------------------
    # Utility Functions
    # -------------------------
    
    
    def fix_frequency_domain(self, image):
        """Fix frequency domain issues (Gaussian high-pass filter)"""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2

        # Create a Gaussian high-pass filter
        x = np.linspace(-ccol, ccol-1, cols)
        y = np.linspace(-crow, crow-1, rows)
        xv, yv = np.meshgrid(x, y)
        sigma = 30  # Adjust sigma for filter strength
        gaussian_filter = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
        mask = 1 - gaussian_filter # Invert to make it high-pass

        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def remove_salt_and_pepper(self, image):
        """Remove salt and pepper noise (Codebase 2 & 3)"""
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        noise_level = np.std(result - image)
        if noise_level > 20:
            result = cv2.medianBlur(result, 3)
        return result

    def fix_blur(self, image):
        """Fix blur (Codebase 1, enhanced with 2-stage approach from Codebase 5)"""
        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Second stage: Sharpening kernel
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        
        # Check if further sharpening is needed
        laplacian_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if laplacian_var < 150:  # Only sharpen more if still somewhat blurry
            return cv2.filter2D(unsharp, -1, kernel)
        else:
            return unsharp

    def fix_contrast(self, image):
        """Fix contrast (Codebase 1, with adaptive CLAHE from Codebase 5)"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # Example parameters
        enhanced = clahe.apply(image)

        # Gamma correction 
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        return enhanced

    def fix_rotation(self, image):
        """Fix rotation (Codebase 1)"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = []
            for rho, theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle-180)
            if angles:
                median_angle = np.median(angles)
                rows, cols = image.shape
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                
                # Calculate padding to avoid cropping after rotation
                angle_rad = np.radians(abs(median_angle))
                
                new_width = int(cols * np.abs(np.cos(angle_rad)) + rows * np.abs(np.sin(angle_rad)))
                new_height = int(rows * np.abs(np.cos(angle_rad)) + cols * np.abs(np.sin(angle_rad)))
                
                
                pad_x = (new_width - cols) // 2
                pad_y = (new_height - rows) // 2

               
                rotated = cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                
                # Add padding to the rotated image
                rotated = cv2.copyMakeBorder(rotated, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255))

                return rotated

        return image

    def binarize(self, image):
        """Binarize the image (Codebase 2's improved binarization)"""
        # Normalize contrast first
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        # Apply adaptive thresholding with larger window
        binary = cv2.adaptiveThreshold(
            normalized, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            81,  # Larger window
            7
        )
        
        # Light cleanup without creating holes
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def estimate_bar_thickness(self, image):
        """Estimate the typical bar thickness (Codebase 7)"""
        # Use a temporary binarization to find runs of black pixels
        temp_binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 10
        )

        # Find vertical black runs (heights of black bars)
        heights = []
        for col in range(temp_binary.shape[1]):
            column = temp_binary[:, col]
            black_runs = []
            run_length = 0
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

    def remove_foreign_object(self, image):
        """Remove foreign objects (Codebase 7's bar thickness logic, enhanced)"""
        gray = image.copy()

        # Estimate bar thickness (after frequency domain and contrast fixes)
        temp_processed = gray.copy()
        
        # Only apply fixes if the problems are detected in the initial problem detection
        if 'frequency_domain' in problems:
            temp_processed = self.fix_frequency_domain(temp_processed)
        if 'low_contrast' in problems:
            temp_processed = self.fix_contrast(temp_processed)
            
        bar_thickness = self.estimate_bar_thickness(temp_processed)

        # Get vertical and horizontal profiles
        vertical_profile = np.mean(gray, axis=0)
        horizontal_profile = np.mean(gray, axis=1)

        # Calculate standard deviations
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)

        # Find anomalous regions
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * v_std
        h_anomalous = np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3.0 * h_std
        
        v_anomalous_regions = self.get_anomalous_regions(np.where(v_anomalous)[0], bar_thickness)
        h_anomalous_regions = self.get_anomalous_regions(np.where(h_anomalous)[0], bar_thickness)

        cleaned = gray.copy()
        
        # --- Vertical pass ---
        for region in v_anomalous_regions:
            for i in region:
                
                
                left_idx = i - 1
                while left_idx >= 0 and left_idx >= i - bar_thickness and v_anomalous[left_idx]:
                    left_idx -= 1

                
                right_idx = i + 1
                while right_idx < gray.shape[1] and right_idx <= i + bar_thickness and v_anomalous[right_idx]:
                    right_idx += 1

               
                if left_idx >= 0 and left_idx >= i - bar_thickness and not v_anomalous[left_idx]:
                    
                    left_bar_values = gray[:, max(0, left_idx - bar_thickness):left_idx + 1]
                    
                    cleaned[:, i] = np.mean(left_bar_values, axis=1)
                elif right_idx < gray.shape[1] and right_idx <= i + bar_thickness and not v_anomalous[right_idx]:
                    
                    right_bar_values = gray[:, right_idx:min(gray.shape[1], right_idx + bar_thickness + 1)]
                    
                    cleaned[:, i] = np.mean(right_bar_values, axis=1)
        
        # --- Horizontal pass ---
        for region in h_anomalous_regions:
            for j in region:
                
                top_idx = j - 1
                while top_idx >= 0 and top_idx >= j - bar_thickness and h_anomalous[top_idx]:
                    top_idx -= 1

                
                bottom_idx = j + 1
                while bottom_idx < cleaned.shape[0] and bottom_idx <= j + bar_thickness and h_anomalous[bottom_idx]:
                    bottom_idx += 1

               
                if top_idx >= 0 and top_idx >= j - bar_thickness and not h_anomalous[top_idx]:
                    
                    top_bar_values = cleaned[max(0, top_idx - bar_thickness):top_idx + 1, :]
                    
                    cleaned[j, :] = np.mean(top_bar_values, axis=0)
                elif bottom_idx < cleaned.shape[0] and bottom_idx <= j + bar_thickness and not h_anomalous[bottom_idx]:
                    
                    bottom_bar_values = cleaned[bottom_idx:min(cleaned.shape[0], bottom_idx + bar_thickness + 1), :]
                    
                    cleaned[j, :] = np.mean(bottom_bar_values, axis=0)

        return cleaned

    def crop_to_barcode(self, binary):
        """Crop the image to the barcode region (Codebase 1 & 2)"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        # Filter contours by aspect ratio and area
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = w * h
            if 1.5 < aspect_ratio < 5 and area > (binary.shape[0] * binary.shape[1] * 0.05):
                valid_contours.append((contour, area))

        if valid_contours:
            largest_contour = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(largest_contour)

            # --- Refined cropping logic ---
            # 1. Analyze horizontal and vertical profiles to find tighter bounds
            horizontal_profile = np.sum(binary, axis=1)
            vertical_profile = np.sum(binary, axis=0)

            # 2. Find the first and last rows/cols with significant white pixel count
            
            
            
            first_row = np.where(horizontal_profile > w * 255 * 0.1)[0][0]
            last_row = np.where(horizontal_profile > w * 255 * 0.1)[0][-1]
            first_col = np.where(vertical_profile > h * 255 * 0.1)[0][0]
            last_col = np.where(vertical_profile > h * 255 * 0.1)[0][-1]

            # 3. Check if these bounds are tighter than the contour bounds
            if first_col > x:
                x = first_col
            if last_col < x + w:
                w = last_col - x
            if first_row > y:
                y = first_row
            if last_row < y + h:
                h = last_row - y
            
            
            if not (w > binary.shape[1] * 0.8 and h > binary.shape[0] * 0.8):
                
                padding_x = int(0.1 * w)
                padding_y = int(0.1 * h)
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w = min(binary.shape[1] - x, w + 2 * padding_x)
                h = min(binary.shape[0] - y, h + 2 * padding_y)

            return binary[y:y+h, x:x+w]

        return binary
    
    def fix_compression_artifacts(self, image):
        """Apply mild noise reduction to fix compression artifacts"""
        return cv2.fastNlMeansDenoising(image, None, 5, 7, 21)

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image
        
        
        global problems 
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        processed = gray.copy()
        
        # --- Apply fixes in optimal order: ---
        
        # 1. Foreign object removal
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed)
            self.add_step("Removed Foreign Object", processed)

        # 2. Frequency domain
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed)
            self.add_step("Fixed Frequency Domain", processed)
            
        # 3. Compression artifacts (before more aggressive noise removal)
        if 'compression_artifacts' in problems:
            processed = self.fix_compression_artifacts(processed)
            self.add_step("Fixed Compression Artifacts", processed)

        # 4. Salt & Pepper
        if 'salt_pepper' in problems:
            processed = self.remove_salt_and_pepper(processed)
            self.add_step("Removed Salt & Pepper", processed)

        # 5. Blur
        if 'blur' in problems:
            processed = self.fix_blur(processed)
            self.add_step("Fixed Blur", processed)

        # 6. Low Contrast
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed)
            self.add_step("Enhanced Contrast", processed)

        # Binarize
        binary = self.binarize(processed)
        self.add_step("Binarized Image", binary)

        # Rotation
        if 'rotation' in problems:
            binary = self.fix_rotation(binary)
            self.add_step("Corrected Rotation", binary)

        # Crop
        cropped = self.crop_to_barcode(binary)
        self.add_step("Cropped Barcode", cropped)

        return cropped

    def visualize_steps(self):
        """Visualize all processing steps"""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (description, img) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(description, fontsize=8)
            ax.axis('off')

        for i in range(len(self.steps), rows*cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    preprocessor = BarcodePreprocessor()
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()