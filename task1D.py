import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.min_bar_width = None
        self.pattern_width = None

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def estimate_bar_thickness(self, image):
        """Estimate bar thickness using gradient analysis"""
        # Normalize and smooth
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

        # Get vertical gradient
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        # Analyze middle portion of image
        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3, 9  # Default fallback

        # Calculate min width and pattern width
        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))

        # Apply safety caps
        min_width = min(min_width, 10)
        pattern_width = min(pattern_width, 30)

        return min_width, pattern_width

    def detect_barcode_region(self, image):
        """Find barcode region in image"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        # Blur and threshold
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter contours by aspect ratio and edge density
        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 10:  # Size threshold
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

    def detect_image_problems(self, image):
        """Detect all image problems"""
        problems = []
        region = self.detect_barcode_region(image)
        if region is None:
            return problems, (0, 0, image.shape[1], image.shape[0])

        x, y, w, h = region
        roi = image[y:y+h, x:x+w]

        # Estimate bar thickness first
        self.min_bar_width, self.pattern_width = self.estimate_bar_thickness(roi)

        # Check blur
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.mean(edges > 0)
        if laplacian_var < 100 or (laplacian_var < 150 and edge_density < 0.15):
            problems.append('blur')

        # Check contrast
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 60:
            problems.append('low_contrast')

        # Check rotation
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(w,h)//3)
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            if angles and abs(np.median(angles)) > 5:
                problems.append('rotation')

        # Check foreign objects using pattern analysis
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        
        v_anomalies = np.where(np.abs(vertical_profile - np.mean(vertical_profile)) > 2.5 * v_std)[0]
        h_anomalies = np.where(np.abs(horizontal_profile - np.mean(horizontal_profile)) > 2.5 * h_std)[0]
        
        if len(v_anomalies) > self.min_bar_width or len(h_anomalies) > self.min_bar_width:
            problems.append('foreign_object')

        # Check salt & pepper noise
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.percentile(noise_diff, 95) > 40:
            problems.append('salt_pepper')

        # Check frequency domain issues
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        return problems, region

    def fix_rotation(self, image, barcode_region):
        """Fix rotation with proper padding and interpolation"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Add padding
        pad = int(max(w, h) * 0.3)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=255)

        # Detect edges and lines
        edges = cv2.Canny(padded, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)

            if angles:
                median_angle = np.median(angles)
                
                # Calculate new dimensions
                angle_rad = abs(np.radians(median_angle))
                rows, cols = padded.shape
                new_w = int(cols * abs(np.cos(angle_rad)) + rows * abs(np.sin(angle_rad)))
                new_h = int(rows * abs(np.cos(angle_rad)) + cols * abs(np.sin(angle_rad)))

                # Adjust rotation matrix
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                matrix[0, 2] += (new_w - cols) / 2
                matrix[1, 2] += (new_h - rows) / 2

                # Perform rotation
                rotated = cv2.warpAffine(padded, matrix, (new_w, new_h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)

                # Find actual content in rotated image
                edges = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges > 0))
                if len(coords) > 0:
                    x_, y_, w_, h_ = cv2.boundingRect(coords)
                    rotated_crop = rotated[y_:y_+h_, x_:x_+w_]
                    
                    # Ensure output size matches input
                    if rotated_crop.shape != (h, w):
                        rotated_crop = cv2.resize(rotated_crop, (w, h))

                    result = image.copy()
                    result[y:y+h, x:x+w] = rotated_crop
                    return result

        return image

    def fix_contrast(self, image, barcode_region):
        """Fix contrast using CLAHE"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Check if extreme low contrast
        if np.std(roi) < 10:
            p2, p98 = np.percentile(roi, (2, 98))
            if p2 == p98:
                p2 = 0
                p98 = 255
            roi = np.interp(roi, (p2, p98), (0, 255)).astype(np.uint8)
        else:
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi = clahe.apply(roi)

        result = image.copy()
        result[y:y+h, x:x+w] = roi
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Fix frequency domain issues using high-pass filter"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2

        # Create high-pass filter mask
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        x_grid, y_grid = np.ogrid[:rows, :cols]
        mask_area = (x_grid - crow)**2 + (y_grid - ccol)**2 <= r*r
        mask[mask_area] = 0

        # Apply filter
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize output
        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def remove_foreign_object(self, image, barcode_region):
        """Remove foreign objects using pattern-aware filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Create vertical averaging kernel
        kernel = np.ones((h, self.min_bar_width)) / h

        # Get intensity profiles
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        
        # Calculate thresholds
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        
        # Detect anomalies
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > 2.5 * v_std
        h_anomalous = np.abs(horizontal_profile - np.mean(horizontal_profile)) > 2.5 * h_std

        cleaned = roi.copy()

        # Process vertical anomalies
        for i in range(w):
            if v_anomalous[i]:
                good_cols = []
                for offset in range(-self.pattern_width, self.pattern_width + 1):
                    check_idx = i + offset
                    if 0 <= check_idx < w and not v_anomalous[check_idx]:
                        good_cols.append(roi[:, check_idx])

                if good_cols:
                    cleaned[:, i] = np.mean(good_cols, axis=0)

        # Process horizontal anomalies
        for j in range(h):
            if h_anomalous[j]:
                good_rows = []
                for offset in range(-self.pattern_width, self.pattern_width + 1):
                    check_idx = j + offset
                    if 0 <= check_idx < h and not h_anomalous[check_idx]:
                        good_rows.append(cleaned[check_idx, :])

                if good_rows:
                    cleaned[j, :] = np.mean(good_rows, axis=0)

        # Final vertical smoothing
        final = cv2.filter2D(cleaned, -1, kernel)

        result = image.copy()
        result[y:y+h, x:x+w] = final
        return result

    def remove_salt_pepper(self, image, barcode_region):
        """Remove salt and pepper noise using progressive median filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Progressive median filtering
        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)

        # Check noise level
        noise_level = np.std(filtered - roi)
        if noise_level > 30:
            filtered = cv2.medianBlur(filtered, 3)

        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def fix_blur(self, image, barcode_region):
        """Fix blur using unsharp masking with adaptive parameters"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # First stage: Unsharp masking with gentle parameters
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.3, gaussian, -0.3, 0)

        # Check if we need more sharpening
        lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if lap_var < 80:
            # Second stage: More aggressive sharpening
            gaussian2 = cv2.GaussianBlur(unsharp, (0,0), 1)
            unsharp = cv2.addWeighted(unsharp, 1.2, gaussian2, -0.2, 0)

        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def binarize(self, image):
        """Binarize using adaptive thresholding with optimized parameters"""
        # Normalize contrast first
        p2, p98 = np.percentile(image, (2, 98))
        if p2 == p98:
            normalized = image
        else:
            normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Window size (should be odd)
            5    # Constant subtracted from mean
        )

        # Light cleanup without creating holes
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def final_cleanup(self, binary):
        """Final cleanup using vertical averaging based on bar pattern"""
        if self.min_bar_width is None:
            self.min_bar_width, _ = self.estimate_bar_thickness(binary)
        
        min_bar_width = max(1, min(self.min_bar_width, 10))
        
        # Create vertical averaging kernel
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]
        
        # Apply kernel along horizontal axis
        cleaned = np.zeros_like(binary)
        for i in range(0, binary.shape[1] - min_bar_width + 1):
            region = binary[:, i:i+min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i+min_bar_width] = (vertical_avg > 127).astype(np.uint8) * 255
        
        return cleaned

    def find_optimal_crop(self, binary):
        """Find optimal crop region that includes barcode and numbers beneath"""
        # Use contours to find main barcode region
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding but keep aspect ratio reasonable
        pad_x = int(0.05 * w)  # 5% horizontal padding
        pad_y = int(0.1 * h)   # 10% vertical padding for numbers
        
        # Ensure we don't exceed image boundaries
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(binary.shape[1] - x, w + 2*pad_x)
        h = min(binary.shape[0] - y, h + 2*pad_y)

        return binary[y:y+h, x:x+w]

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Detect problems and barcode region
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if barcode_region is None:
            return gray

        processed = gray.copy()

        # Fix rotation first if needed
        if 'rotation' in problems:
            processed = self.fix_rotation(processed, barcode_region)
            self.add_step("Fixed Rotation", processed)
            
            # Re-detect barcode region after rotation
            new_region = self.detect_barcode_region(processed)
            if new_region is not None:
                barcode_region = new_region
            else:
                # If we lost the region, revert rotation
                processed = gray.copy()
                barcode_region = self.detect_barcode_region(processed)

        # Fix frequency domain issues next
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed, barcode_region)
            self.add_step("Fixed Frequency Domain", processed)

        # Then handle foreign objects
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed, barcode_region)
            self.add_step("Removed Foreign Object", processed)

        # Fix noise issues
        if 'salt_pepper' in problems:
            processed = self.remove_salt_pepper(processed, barcode_region)
            self.add_step("Removed Salt & Pepper", processed)

        # Fix blur if present
        if 'blur' in problems:
            processed = self.fix_blur(processed, barcode_region)
            self.add_step("Fixed Blur", processed)

        # Fix contrast last before binarization
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed, barcode_region)
            self.add_step("Enhanced Contrast", processed)

        # Extract ROI for final processing
        x, y, w, h = barcode_region
        roi = processed[y:y+h, x:x+w]

        # Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        # Final cleanup using vertical averaging
        cleaned = self.final_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        # Find optimal crop
        final = self.find_optimal_crop(cleaned)
        self.add_step("Final Result", final)

        return final

    def visualize_steps(self):
        """Visualize all processing steps"""
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