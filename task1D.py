import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.min_bar_width = None
        self.pattern_width = None

    def add_step(self, description, image):
        self.steps.append((description, image.copy()))

    def estimate_bar_thickness(self, image):
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 1, 3

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))

        return min(min_width, 3), min(pattern_width, 9)

    def detect_barcode_region(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 10:
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 7 and edge_density > 0.1:  # Increased aspect ratio
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        # Sort by area and pick the largest
        valid_regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        return valid_regions[0] if valid_regions else None

    def detect_image_problems(self, image):
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]

        # Get pattern info first
        self.min_bar_width, self.pattern_width = self.estimate_bar_thickness(roi)

        # Foreign object detection
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)
        if np.any(np.abs(vertical_profile - v_mean) > 2.5 * v_std):
            problems.append('foreign_object')

        # Rotation detection (do this early)
        edges = cv2.Canny(roi, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            try:
                angle = cv2.minAreaRect(coords)[-1]
                if abs(angle) > 5:
                    problems.append('rotation')
            except:
                pass # Handle cases where minAreaRect fails

        # Blur detection
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        if laplacian_var < 80:
            problems.append('blur')

        # Low contrast
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 50:
            problems.append('low_contrast')

        # Salt and pepper
        median_blurred = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median_blurred)
        if np.mean(noise_diff) > 10:
            problems.append('salt_pepper')

        # Frequency domain
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        return problems, barcode_region

    def remove_foreign_object(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Get vertical profile
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)

        # Get smallest bar width
        min_bar_width, _ = self.estimate_bar_thickness(roi)
        min_bar_width = min(min_bar_width, 3)

        # Detect anomalies
        anomalies = np.abs(vertical_profile - v_mean) > 2.5 * v_std

        # Apply vertical averaging only on anomalous regions
        cleaned = roi.copy()
        for i in range(w):
            if anomalies[i]:
                col = roi[:, i]
                cleaned[:, i] = np.mean(col) * np.ones(h)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def remove_salt_pepper(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Apply a bilateral filter first to smooth while preserving edges
        bilateral = cv2.bilateralFilter(roi, 9, 75, 75)

        # Then apply a median filter
        result = cv2.medianBlur(bilateral, 3)

        result_img = image.copy()
        result_img[y:y+h, x:x+w] = result
        return result_img

    def fix_rotation(self, image):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)  # Reduced threshold
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if 40 < angle < 140: # Consider only horizontal/vertical lines
                    if angle < 90:
                        angles.append(angle)
                    else:
                        angles.append(angle - 180)
            if angles:
                median_angle = np.median(angles)
                rows, cols = image.shape
                matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)
                rotated = cv2.warpAffine(image, matrix, (cols, rows),
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REPLICATE) # Use replicate
                return rotated
        return image

    def fix_contrast(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)

        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def fix_blur(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Apply a mild sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(roi, -1, kernel)

        return sharpened

    def fix_frequency_domain(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2

        # Create a high-pass filter mask (emphasize edges)
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 40, 1, -1) # Keep low frequencies
        mask = 1 - mask

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def binarize(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0) # Apply slight blur before thresholding
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def find_optimal_crop(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:min(5, len(contours))]: # Check top 5 contours
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 5 and 1.0 <= w / h <= 8.0: # Basic filtering
                pad_y_top = int(0.1 * h)
                pad_y_bottom = int(0.2 * h)
                pad_x = int(0.1 * w)

                x = max(0, x - pad_x)
                y = max(0, y - pad_y_top)
                w = min(binary.shape[1] - x, w + 2 * pad_x)
                h = min(binary.shape[0] - y, h + pad_y_top + pad_y_bottom)
                return binary[y:y + h, x:x + w]

        return binary # If no suitable contour found

    def final_cleanup(self, binary):
        if np.sum(binary) == 0:  # Handle completely black images
            return binary
        min_width = 1
        h, w = binary.shape
        cleaned = np.zeros_like(binary)
        for i in range(w):
            col = binary[:, i]
            cleaned[:, i] = (np.mean(col) > 127) * 255
        return cleaned

    def preprocess_image(self, image):
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if barcode_region is None:
            return gray

        processed = gray.copy()
        x, y, w, h = barcode_region

        # Fix rotation first if needed
        if 'rotation' in problems:
            processed = self.fix_rotation(processed)
            self.add_step("Fixed Rotation", processed)

            # Re-detect barcode region
            new_region = self.detect_barcode_region(processed)
            if new_region:
                barcode_region = new_region
                x, y, w, h = barcode_region
            else:
                processed = gray.copy()
                barcode_region = self.detect_barcode_region(processed)
                if barcode_region:
                    x, y, w, h = barcode_region

        # Extract ROI
        roi = processed[y:y+h, x:x+w].copy()

        # Remove foreign objects first
        if 'foreign_object' in problems:
            roi = self.remove_foreign_object(roi, (0, 0, w, h))
            self.add_step("Removed Foreign Object", roi)

        # Fix salt & pepper next
        if 'salt_pepper' in problems:
            roi = self.remove_salt_pepper(roi, (0, 0, w, h))
            self.add_step("Removed Salt & Pepper", roi)

        # Fix frequency domain issues
        if 'frequency_domain' in problems:
            roi = self.fix_frequency_domain(roi, (0, 0, w, h))
            self.add_step("Fixed Frequency Domain", roi)

        # Fix blur
        if 'blur' in problems:
            roi = self.fix_blur(roi, (0, 0, w, h))
            self.add_step("Fixed Blur", roi)

        # Fix contrast last before binarization
        if 'low_contrast' in problems:
            roi = self.fix_contrast(roi, (0, 0, w, h))
            self.add_step("Enhanced Contrast", roi)

        # Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        # Find optimal crop
        cropped = self.find_optimal_crop(binary)
        self.add_step("Cropped", cropped)

        # Final vertical averaging filter
        final = self.final_cleanup(cropped)
        self.add_step("Final Cleanup", final)

        return final

    def visualize_steps(self):
        n_steps = len(self.steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

        # Handle single image case
        if rows * cols == 1:
            axes = np.array([axes])

        # Reshape axes for proper indexing
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

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

        # Hide empty subplots
        for i in range(len(self.steps), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

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
        else:
            print("Failed to load image.")