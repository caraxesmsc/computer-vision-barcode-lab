import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.problems = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))
        
    def estimate_bar_thickness(self, image):
        """Estimate barcode bar thickness for pattern-aware processing"""
        # Initial preprocessing to improve estimation
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

        # Get vertical gradient
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        # Find typical bar width using middle portion
        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3, 9  # Default fallback

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        
        # Sanity check on returned values
        min_width = min(min_width, 10)  # Cap maximum width
        pattern_width = min(pattern_width, 30)  # Cap maximum pattern width

        return min_width, pattern_width

    def get_anomalous_regions(self, anomalies, bar_thickness):
        """Group contiguous anomalous regions based on bar thickness"""
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

    def detect_barcode_region(self, image):
        """Find the actual barcode region in the image"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        # Find regions with high concentration of vertical lines
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find connected components
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by aspect ratio and vertical line density
        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:  # Too small
                continue

            aspect_ratio = w / float(h)
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)

            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        # Return region with highest edge density
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]))

    def detect_compression_artifacts(self, gray):
        """Detect compression artifacts using edge analysis"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        block_size = 8
        blurred = cv2.boxFilter(magnitude.astype(np.float32), -1, (block_size, block_size))
        block_variation = np.std(blurred)
        return block_variation > 10

    def detect_image_problems(self, image):
        """Detect image problems"""
        problems = []
        self.problems = []  # Reset problems list
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Find barcode region
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]

        # Get bar pattern info
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)

        # Foreign object detection
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)

        v_anomalies = np.where(np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * v_std)[0]
        h_anomalies = np.where(np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3.0 * h_std)[0]

        v_anomalous_regions = self.get_anomalous_regions(v_anomalies, min_bar_width)
        h_anomalous_regions = self.get_anomalous_regions(h_anomalies, min_bar_width)

        if len(v_anomalous_regions) > 0 or len(h_anomalous_regions) > 0:
            problems.append('foreign_object')

        # Blur detection
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.mean(edges > 0)
        
        if laplacian_var < 90 and edge_density < 0.15:
            problems.append('blur')

        # Contrast detection
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')

        # Salt and pepper detection
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.percentile(noise_diff, 95) > 40:
            problems.append('salt_pepper')

        # Compression artifacts detection
        if self.detect_compression_artifacts(roi):
            problems.append('compression_artifacts')

        # Frequency domain detection
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # Rotation detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(w,h)//3)
        if lines is not None:
            angles = []
            for rho, theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            if angles and abs(np.median(angles)) > 5:
                problems.append('rotation')

        self.problems = problems.copy()
        return problems, barcode_region

    def remove_foreign_object(self, image, barcode_region):
        """Remove foreign objects using pattern-aware vertical filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Get bar pattern info
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)

        # Create vertical averaging kernel
        kernel_height = h
        kernel_width = min_bar_width
        kernel = np.ones((kernel_height, kernel_width)) / kernel_height

        # Get vertical profile
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)

        # Calculate thresholds
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)

        # Detect anomalies
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * v_std
        h_anomalous = np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3.0 * h_std

        cleaned = roi.copy()

        # Process vertical anomalies
        for i in range(w):
            if v_anomalous[i]:
                # Look for good columns within pattern_width
                good_cols = []
                for offset in range(-pattern_width, pattern_width + 1):
                    check_idx = i + offset
                    if 0 <= check_idx < w and not v_anomalous[check_idx]:
                        good_cols.append(roi[:, check_idx])

                if good_cols:
                    cleaned[:, i] = np.mean(good_cols, axis=0)

        # Process horizontal anomalies
        for j in range(h):
            if h_anomalous[j]:
                good_rows = []
                for offset in range(-pattern_width, pattern_width + 1):
                    check_idx = j + offset
                    if 0 <= check_idx < h and not h_anomalous[check_idx]:
                        good_rows.append(cleaned[check_idx, :])

                if good_rows:
                    cleaned[j, :] = np.mean(good_rows, axis=0)

        # Final pass: Apply vertical averaging kernel
        filtered = cv2.filter2D(cleaned, -1, kernel)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Fix frequency domain issues using high-pass filter"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        x_grid, y_grid = np.ogrid[:rows, :cols]
        mask_area = (x_grid - crow)**2 + (y_grid - ccol)**2 <= r*r
        mask[mask_area] = 0

        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_blur(self, image, barcode_region):
        """Two-stage deblurring"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.3, gaussian, -0.3, 0)

        # Second stage: Only if needed
        lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if lap_var < 80:
            gaussian2 = cv2.GaussianBlur(unsharp, (0,0), 1)
            unsharp = cv2.addWeighted(unsharp, 1.2, gaussian2, -0.2, 0)

        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def fix_compression_artifacts(self, image, barcode_region):
        """Fix compression artifacts using non-local means denoising"""
        x, y, w, h = barcode_region
        roi = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply non-local means denoising with careful parameters
        denoised = cv2.fastNlMeansDenoising(
            roi,
            None,
            h=5,  # Filter strength - lower than usual to preserve barcode details
            templateWindowSize=7,
            searchWindowSize=21
        )

        return denoised

    def fix_contrast(self, image, barcode_region):
        """Fix contrast using CLAHE"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)

        # Preserve black and white points
        enhanced[roi < 30] = 0
        enhanced[roi > 225] = 255

        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def remove_salt_pepper(self, image, barcode_region):
        """Remove salt and pepper noise"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)

        # Extra pass if needed
        noise_level = np.std(filtered - roi)
        if noise_level > 30:
            filtered = cv2.medianBlur(filtered, 3)

        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def fix_rotation(self, image, barcode_region):
        """Fix rotation with proper padding"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Add padding
        pad = int(max(w, h) * 0.3)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

        # Detect edges and find rotation angle
        edges = cv2.Canny(padded, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

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
                rows, cols = padded.shape
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)

                # Calculate new dimensions
                angle_rad = abs(np.radians(median_angle))
                new_width = int(cols * abs(np.cos(angle_rad)) + rows * abs(np.sin(angle_rad)))
                new_height = int(rows * abs(np.cos(angle_rad)) + cols * abs(np.sin(angle_rad)))

                # Adjust rotation matrix
                matrix[0, 2] += (new_width - cols) / 2
                matrix[1, 2] += (new_height - rows) / 2

                # Perform rotation
                rotated = cv2.warpAffine(padded, matrix, (new_width, new_height),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)

                # Find actual barcode region in rotated image
                edges = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges > 0))
                if len(coords) > 0:
                    x_, y_, w_, h_ = cv2.boundingRect(coords)

                    # Ensure dimensions don't exceed original
                    if w_ > w or h_ > h:
                        scale = min(w/w_, h/h_)
                        w_ = int(w_ * scale)
                        h_ = int(h_ * scale)

                    # Ensure valid coordinates
                    x_ = max(0, min(x_, rotated.shape[1] - w_))
                    y_ = max(0, min(y_, rotated.shape[0] - h_))

                    rotated_crop = rotated[y_:y_+h_, x_:x_+w_]

                    # Resize if necessary
                    if rotated_crop.shape != (h, w):
                        rotated_crop = cv2.resize(rotated_crop, (w, h))

                    result = image.copy()
                    result[y:y+h, x:x+w] = rotated_crop
                    return result

        return image

    def binarize(self, image):
        """Binarize using adaptive thresholding"""
        # Normalize contrast
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # window size
            5    # constant subtracted from mean
        )

        # Light cleanup
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def final_cleanup(self, binary):
        """Final cleanup using vertical averaging"""
        min_bar_width, _ = self.estimate_bar_thickness(binary)  # Fixed to unpack tuple
        min_bar_width = max(1, min(min_bar_width, 10))  # Ensure reasonable value

        # Create vertical averaging kernel
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]

        # Apply kernel
        cleaned = np.zeros_like(binary)
        for i in range(0, binary.shape[1] - min_bar_width + 1):
            region = binary[:, i:i+min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i+min_bar_width] = (vertical_avg > 127).astype(np.uint8) * 255

        return cleaned

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Detect problems
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if barcode_region is None:
            return gray

        processed = gray.copy()
        x, y, w, h = barcode_region

        # Fix rotation first
        if 'rotation' in problems:
            processed = self.fix_rotation(processed, barcode_region)
            self.add_step("Fixed Rotation", processed)
            
            # Update barcode region
            new_region = self.detect_barcode_region(processed)
            if new_region:
                barcode_region = new_region
                x, y, w, h = barcode_region

        # Process ROI
        roi = processed[y:y+h, x:x+w]

        # Apply fixes in optimal order
        if 'foreign_object' in problems:
            roi = self.remove_foreign_object(roi, (0, 0, w, h))
            self.add_step("Removed Foreign Object", roi)

        if 'frequency_domain' in problems:
            roi = self.fix_frequency_domain(roi, (0, 0, w, h))
            self.add_step("Fixed Frequency Domain", roi)

        if 'compression_artifacts' in problems:
            roi = self.fix_compression_artifacts(roi, (0, 0, w, h))
            self.add_step("Fixed Compression Artifacts", roi)

        if 'salt_pepper' in problems:
            roi = self.remove_salt_pepper(roi, (0, 0, w, h))
            self.add_step("Removed Salt & Pepper", roi)

        if 'blur' in problems:
            roi = self.fix_blur(roi, (0, 0, w, h))
            self.add_step("Fixed Blur", roi)

        if 'low_contrast' in problems:
            roi = self.fix_contrast(roi, (0, 0, w, h))
            self.add_step("Enhanced Contrast", roi)

        # Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        # Final cleanup
        cleaned = self.final_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        return cleaned

    def visualize_steps(self):
        """Visualize processing steps"""
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