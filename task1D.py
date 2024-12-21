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

        # Find typical bar width using middle portion of image
        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3, 9  # Default fallback

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        
        return min_width, pattern_width

    def detect_barcode_region(self, image):
        """Find the actual barcode region in the image"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        # Find regions with high concentration of vertical lines
        blur = cv2.GaussianBlur(gradx, (21,21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
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
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50) / (w * h)
            
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))
        
        if not valid_regions:
            return None
            
        # Return region with highest edge density
        x, y, w, h = max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))
        
        # Add padding
        pad_x = int(0.1 * w)
        pad_y = int(0.1 * h)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(image.shape[1] - x, w + 2*pad_x)
        h = min(image.shape[0] - y, h + 2*pad_y)
        
        return x, y, w, h

    def detect_image_problems(self, image, barcode_region=None):
        """Detect image problems with explicit handling of region absence"""
        problems = []
        gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])  # Use image dimensions as a fallback

        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]

        # Estimate bar width and pattern width with added check
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)
        if min_bar_width == 0 or pattern_width == 0:
            return problems, barcode_region

        # Foreign object detection
        if self.detect_foreign_object(roi, min_bar_width, pattern_width):
            problems.append('foreign_object')

        # Blur detection
        if self.detect_blur(roi):
            problems.append('blur')

        # Contrast detection
        if self.detect_low_contrast(roi):
            problems.append('low_contrast')

        # Salt and pepper detection
        if self.detect_salt_pepper(roi):
            problems.append('salt_pepper')

        # Frequency domain analysis
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 65:
            problems.append('frequency_domain')

        # Rotation detection
        if self.detect_rotation(roi):
            problems.append('rotation')

        self.problems = problems
        return problems, barcode_region

    def detect_foreign_object(self, roi, min_bar_width, pattern_width):
        """Detect foreign objects using pattern analysis and a rolling window standard deviation."""
        vertical_profile = np.mean(roi, axis=0)
        
        # Calculate rolling standard deviation with a window size relative to the pattern width
        window_size = pattern_width
        if window_size > len(vertical_profile):
            return False
        
        v_stds = [np.std(vertical_profile[i:i+window_size]) for i in range(len(vertical_profile)-window_size + 1)]
        
        v_std_anomalies = np.where(np.abs(v_stds - np.mean(v_stds)) > 2.5 * np.std(v_stds))[0]
        
        if len(v_std_anomalies) > 0:
            return True

        return False

    def detect_blur(self, roi):
        """Detect blur using Laplacian variance and frequency analysis."""
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        if laplacian_var < 100:
            return True  # Blur likely

        # Frequency domain analysis for blur
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        rows, cols = roi.shape
        crow, ccol = rows // 2, cols // 2
        low_freq_magnitude = np.sum(magnitude_spectrum[crow-5:crow+5, ccol-5:ccol+5])
        high_freq_magnitude = np.sum(magnitude_spectrum) - low_freq_magnitude
        
        # Avoid division by zero
        freq_ratio = high_freq_magnitude / low_freq_magnitude if low_freq_magnitude != 0 else 0

        # Combine metrics
        blur_score = laplacian_var * freq_ratio
        return blur_score < 7  # Adjust threshold as needed

    def detect_low_contrast(self, roi):
        """Detect low contrast using percentiles."""
        p5, p95 = np.percentile(roi, [5, 95])
        return (p95 - p5) < 80  # Adjust threshold as needed

    def remove_salt_and_pepper(self, image, barcode_region):
        """Remove salt and pepper noise using median filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Apply median filtering
        filtered = cv2.medianBlur(roi, 3)

        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def fix_blur(self, image, barcode_region):
        """Fix blur using unsharp masking"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.5, gaussian, -0.5, 0)

        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def fix_contrast(self, image, barcode_region):
        """Fix contrast using CLAHE"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)

        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result
    
    def detect_salt_pepper(self, roi):
        """Detect salt and pepper noise with a stricter threshold."""
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        return np.percentile(noise_diff, 95) > 50  # Increased threshold for strictness

    def get_anomalous_regions(self, indices, min_width):
        """Group anomalous indices into regions based on minimum width"""
        if len(indices) == 0:
            return []
            
        regions = []
        current_region = [indices[0]]
        
        for idx in indices[1:]:
            if idx - current_region[-1] <= min_width:
                current_region.append(idx)
            else:
                if len(current_region) >= min_width:
                    regions.append(current_region)
                current_region = [idx]
                
        if len(current_region) >= min_width:
            regions.append(current_region)
            
        return regions

    def detect_rotation(self, image):
        """Detect if the image needs rotation correction"""
        angle = self.check_rotation_angle(image)
        return abs(angle) > 3

    def detect_compression_artifacts(self, gray):
        """Detect compression artifacts using edge analysis and block-based standard deviation."""
        # Apply Sobel filters to detect edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        block_size = 8
        if magnitude.shape[0] < block_size or magnitude.shape[1] < block_size:
            return False  # Image is too small for block analysis

        blocks_shape = (magnitude.shape[0]//block_size, magnitude.shape[1]//block_size, block_size, block_size)
        blocks = np.lib.stride_tricks.as_strided(magnitude, shape=blocks_shape, strides=magnitude.strides * 2)
        block_stds = np.std(blocks, axis=(2, 3))
        
        return np.std(block_stds) > 10  # Adjust threshold as needed

    def remove_foreign_object(self, image, barcode_region):
        """Remove foreign objects using pattern-aware vertical filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y + h, x:x + w].copy()
        
        if 'foreign_object' not in self.problems:
            return image

        # Estimate bar thickness
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)
        min_bar_width = max(1, min(min_bar_width, 10))

        # Create a vertical averaging kernel
        kernel_height = h
        kernel_width = min_bar_width
        kernel = np.ones((kernel_height, kernel_width)) / kernel_height

        # Calculate vertical and horizontal profiles
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)

        # Calculate standard deviations
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)

        # Set thresholds for anomaly detection (adaptive)
        v_threshold = np.mean(vertical_profile) + 2.5 * v_std
        h_threshold = np.mean(horizontal_profile) + 2.5 * h_std

        # Detect anomalous regions based on vertical profile
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > v_threshold
        h_anomalous = np.abs(horizontal_profile - np.mean(horizontal_profile)) > h_threshold

        v_anomalous_regions = self.get_anomalous_regions(np.where(v_anomalous)[0], min_bar_width)
        h_anomalous_regions = self.get_anomalous_regions(np.where(h_anomalous)[0], min_bar_width)

        cleaned = roi.copy()

        # Process vertical anomalies
        for region in v_anomalous_regions:
            for i in region:
                # Calculate the valid range for averaging
                start = max(0, i - pattern_width)
                end = min(w, i + pattern_width + 1)
                
                # Select the columns for averaging
                good_cols = [roi[:, j] for j in range(start, end) if not v_anomalous[j]]

                # Replace anomalous column with the average of good neighbors
                if good_cols:
                    avg_col = np.mean(good_cols, axis=0)
                    cleaned[:, i:i+1] = avg_col.reshape(-1, 1)

        # Process horizontal anomalies (similar logic as vertical)
        for region in h_anomalous_regions:
            for j in region:
                top_idx = j - 1
                while top_idx >= 0 and top_idx >= j - pattern_width and h_anomalous[top_idx]:
                    top_idx -= 1
                
                bottom_idx = j + 1
                while bottom_idx < h and bottom_idx <= j + pattern_width and h_anomalous[bottom_idx]:
                    bottom_idx += 1

                if top_idx >= 0 and top_idx >= j - pattern_width and not h_anomalous[top_idx]:
                    top_values = cleaned[max(0, top_idx - min_bar_width):top_idx + 1, :]
                    cleaned[j, :] = np.mean(top_values, axis=0)
                elif bottom_idx < h and bottom_idx <= j + pattern_width and not h_anomalous[bottom_idx]:
                    bottom_values = cleaned[bottom_idx:min(h, bottom_idx + min_bar_width + 1), :]
                    cleaned[j, :] = np.mean(bottom_values, axis=0)

        # Apply the vertical kernel for smoothing
        smoothed = cv2.filter2D(cleaned, -1, kernel)

        result = image.copy()
        result[y:y + h, x:x + w] = smoothed
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Fix frequency domain issues using a high-pass filter"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = roi.shape
        crow, ccol = rows // 2, cols // 2
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
        result[y:y + h, x:x + w] = cleaned
        return result
    
    def fix_compression_artifacts(self, image, barcode_region):
        """Fix compression artifacts using non-local means denoising"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        denoised = cv2.fastNlMeansDenoising(
            roi,
            None,
            h=5,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        result = image.copy()
        result[y:y+h, x:x+w] = denoised
        return result

    def check_rotation_angle(self, image):
        """Precisely measure rotation angle using Hough lines"""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                return median_angle
        return 0

    def fix_rotation(self, image, barcode_region):
        """Fix rotation with proper padding and accurate cropping"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        angle = self.check_rotation_angle(roi)
        if abs(angle) <= 3:  # Increased threshold
            return image

        # Add padding before rotation
        pad = int(max(w, h) * 0.3)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

        rows, cols = padded.shape
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        # Calculate new dimensions post-rotation
        angle_rad = np.abs(np.radians(angle))
        new_w = int(cols * np.abs(np.cos(angle_rad)) + rows * np.abs(np.sin(angle_rad)))
        new_h = int(rows * np.abs(np.cos(angle_rad)) + cols * np.abs(np.sin(angle_rad)))

        # Adjust rotation matrix to keep image within bounds
        matrix[0, 2] += (new_w - cols) / 2
        matrix[1, 2] += (new_h - rows) / 2

        # Perform rotation
        rotated = cv2.warpAffine(padded, matrix, (new_w, new_h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)

        # Find actual barcode region in rotated image
        edges = cv2.Canny(rotated, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            x_, y_, w_, h_ = cv2.boundingRect(coords)
            
            # Ensure that the new region does not exceed the bounds of the rotated image
            x_ = max(0, min(x_, rotated.shape[1] - w_))
            y_ = max(0, min(y_, rotated.shape[0] - h_))

            rotated_crop = rotated[y_:y_ + h_, x_:x_ + w_]

            # Resize rotated_crop to match original ROI's w, h
            rotated_crop = cv2.resize(rotated_crop, (w, h))

            result = image.copy()
            result[y:y+h, x:x+w] = rotated_crop
            return result

        return image

    def binarize(self, image):
        """Binarize using adaptive thresholding with a morphological operation."""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 10)
        
        # Use morphological closing to potentially fill gaps and reduce noise
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return closing

    def final_barcode_cleanup(self, binary):
        """Final cleanup of the barcode using vertical averaging"""
        # Use the previously defined function to estimate bar thickness
        min_bar_width, _ = self.estimate_bar_thickness(binary)
        min_bar_width = max(1, min(min_bar_width, 10))  # Ensure reasonable value

        # Create a vertical averaging kernel
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]

        # Apply the kernel along the horizontal axis
        cleaned = cv2.filter2D(binary, -1, kernel)

        return cleaned

    def initial_cleanup(self, image):
        """Basic image enhancement - Applied to all images"""
        denoised = cv2.medianBlur(image, 3)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        
        _, normalized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return normalized

    def preprocess_image(self, image):
        """Main preprocessing pipeline with improved structure"""
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Stage 1: General Enhancement
        enhanced = self.initial_cleanup(gray)
        self.add_step("Initial Enhancement", enhanced)
        
        # Detect barcode region
        barcode_region = self.detect_barcode_region(enhanced)
        if barcode_region is None:
            return gray
        
        x, y, w, h = barcode_region
        roi = enhanced[y:y+h, x:x+w]

        # Stage 2: Core Problems
        # Check and fix rotation first
        angle = self.check_rotation_angle(roi)
        if abs(angle) > 3:
            roi = self.fix_rotation(roi, (0, 0, w, h))
            self.add_step("Fixed Rotation", roi)
        
        # Stage 3: Barcode-Specific Processing
        # Detect remaining problems within the ROI
        problems, _ = self.detect_image_problems(roi, (0, 0, w, h))

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
            roi = self.remove_salt_and_pepper(roi, (0, 0, w, h))
            self.add_step("Removed Salt & Pepper", roi)

        if 'blur' in problems:
            roi = self.fix_blur(roi, (0, 0, w, h))
            self.add_step("Fixed Blur", roi)

        if 'low_contrast' in problems:
            roi = self.fix_contrast(roi, (0, 0, w, h))
            self.add_step("Enhanced Contrast", roi)

        # Stage 4: Final Cleanup
        # Binarize after all other fixes
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        cleaned = self.final_barcode_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        # Update the processed image with the cleaned ROI
        processed = gray.copy()
        processed[y:y+h, x:x+w] = cleaned

        return processed

    def visualize_steps(self):
        """Visualize all processing steps"""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (description, image) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.imshow(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
            ax.set_title(description, fontsize=8)
            ax.axis('off')
            
        # Hide empty subplots
        for i in range(len(self.steps), rows*cols):
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