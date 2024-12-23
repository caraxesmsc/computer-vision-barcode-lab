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
            return 1, 3  # Default to smallest possible bar width

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        
        min_width = min(min_width, 10)
        pattern_width = min(pattern_width, 30)

        return min_width, pattern_width

    def get_anomalous_regions(self, anomalies, bar_thickness):
        """Group contiguous anomalous indices into regions"""
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
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        # Increase blur kernel for better region detection
        blur = cv2.GaussianBlur(gradx, (25, 25), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 10:  # Increased minimum width
                continue
            
            aspect_ratio = w / float(h)
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h, edge_density))

        if not valid_regions:
            return None

        # Return region with highest edge density
        return tuple(max(valid_regions, key=lambda r: r[4])[:4])

    def detect_image_problems(self, image):
        """Detect image problems with improved accuracy"""
        problems = []
        self.problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

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

        # Blur detection with improved threshold
        edges = cv2.Canny(roi, 50, 150)
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edge_density = np.mean(edges > 0)
        
        if laplacian_var < 150 and edge_density < 0.2:  # More sensitive blur detection
            problems.append('blur')

        # Low contrast detection
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 50:
            problems.append('low_contrast')

        # Salt and pepper detection using multi-scale analysis
        median3 = cv2.medianBlur(roi, 3)
        median5 = cv2.medianBlur(roi, 5)
        noise_diff3 = np.abs(roi - median3)
        noise_diff5 = np.abs(roi - median5)
        
        if np.mean(noise_diff3) > 10 or np.mean(noise_diff5) > 15:
            problems.append('salt_pepper')

        # Rotation detection
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        # Frequency domain detection
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        return problems, barcode_region

    def remove_foreign_object(self, image, barcode_region):
        """Remove foreign objects using 1-pixel or minimum bar width vertical averaging"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Try to get bar thickness, fallback to 1 if detection fails
        try:
            min_bar_width, _ = self.estimate_bar_thickness(roi)
            kernel_width = min(min_bar_width, 3)  # Limit kernel width
        except:
            kernel_width = 1
            
        # Create vertical averaging kernel
        kernel = np.ones((h, kernel_width)) / h
        
        # Apply kernel along horizontal axis
        cleaned = np.zeros_like(roi)
        for i in range(0, w - kernel_width + 1):
            region = roi[:, i:i+kernel_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i+kernel_width] = vertical_avg.reshape(-1, 1)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def remove_salt_pepper(self, image, barcode_region):
        """Enhanced salt and pepper removal with progressive filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Try different kernel sizes and combinations
        results = []
        
        # Single kernel attempts
        results.append(('3x3', cv2.medianBlur(roi, 3)))
        results.append(('5x5', cv2.medianBlur(roi, 5)))
        results.append(('7x7', cv2.medianBlur(roi, 7)))
        
        # Combined kernel attempts
        temp = cv2.medianBlur(roi, 3)
        results.append(('3x3+5x5', cv2.medianBlur(temp, 5)))
        
        temp = cv2.medianBlur(roi, 5)
        results.append(('5x5+3x3', cv2.medianBlur(temp, 3)))
        
        # Evaluate results based on edge preservation
        best_score = float('-inf')
        best_result = None
        
        for name, result in results:
            edges = cv2.Canny(result, 50, 150)
            edge_score = np.sum(edges)
            if edge_score > best_score:
                best_score = edge_score
                best_result = result
        
        result = image.copy()
        result[y:y+h, x:x+w] = best_result
        return result

    def fix_rotation(self, image):
        """Fix rotation with proper padding"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
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
                rows, cols = image.shape
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                
                # Calculate new dimensions to avoid cropping
                angle_rad = abs(np.radians(median_angle))
                new_width = int(cols * np.abs(np.cos(angle_rad)) + rows * np.abs(np.sin(angle_rad)))
                new_height = int(rows * np.abs(np.cos(angle_rad)) + cols * np.abs(np.sin(angle_rad)))
                
                # Adjust transformation matrix
                matrix[0, 2] += (new_width - cols) / 2
                matrix[1, 2] += (new_height - rows) / 2
                
                return cv2.warpAffine(image, matrix, (new_width, new_height),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=255)
        return image

    def fix_contrast(self, image, barcode_region):
        """Fix contrast using CLAHE"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        
        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def fix_blur(self, image, barcode_region):
        """Fix blur using unsharp masking with careful parameters"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Gentle unsharp masking
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.3, gaussian, -0.3, 0)
        
        # Check if more sharpening needed
        lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if lap_var < 80:
            gaussian2 = cv2.GaussianBlur(unsharp, (0,0), 1)
            unsharp = cv2.addWeighted(unsharp, 1.2, gaussian2, -0.2, 0)
            
        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Fix frequency domain issues"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2

        # Create high-pass filter
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def binarize(self, image):
        """Careful binarization with pre-normalization"""
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        binary = cv2.adaptiveThreshold(
            normalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Window size
            7    # C value
        )
        
        # Very light morphological cleanup
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def final_cleanup(self, binary):
        """Final vertical averaging cleanup using bar width"""
        try:
            min_bar_width, _ = self.estimate_bar_thickness(binary)
            kernel_width = min(min_bar_width, 3)  # Limit kernel width
        except:
            kernel_width = 1
            
        kernel = np.ones((binary.shape[0], kernel_width)) / binary.shape[0]
        
        # Apply vertical averaging
        cleaned = np.zeros_like(binary)
        for i in range(0, binary.shape[1] - kernel_width + 1):
            region = binary[:, i:i+kernel_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i+kernel_width] = (vertical_avg > 127).astype(np.uint8) * 255
            
        return cleaned

    def find_optimal_crop(self, binary):
        """Find optimal crop that includes barcode and numbers beneath"""
        # Get contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add padding for numbers beneath
        pad_y_top = int(0.05 * h)  # 5% top padding
        pad_y_bottom = int(0.15 * h)  # 15% bottom padding for numbers
        pad_x = int(0.05 * w)  # 5% side padding
        
        # Calculate new coordinates with padding
        x = max(0, x - pad_x)
        y = max(0, y - pad_y_top)
        w = min(binary.shape[1] - x, w + 2*pad_x)
        h = min(binary.shape[0] - y, h + pad_y_top + pad_y_bottom)
        
        return binary[y:y+h, x:x+w]

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if barcode_region is None:
            return gray

        processed = gray.copy()
        x, y, w, h = barcode_region

        # Fix rotation first
        if 'rotation' in problems:
            processed = self.fix_rotation(processed)
            self.add_step("Fixed Rotation", processed)
            
            # Re-detect barcode region
            new_region = self.detect_barcode_region(processed)
            if new_region:
                barcode_region = new_region
                x, y, w, h = barcode_region

        # Extract ROI
        roi = processed[y:y+h, x:x+w]

        # Fix frequency domain issues
        if 'frequency_domain' in problems:
            roi = self.fix_frequency_domain(roi, (0, 0, w, h))
            self.add_step("Fixed Frequency Domain", roi)

        # Remove foreign objects
        if 'foreign_object' in problems:
            roi = self.remove_foreign_object(roi, (0, 0, w, h))
            self.add_step("Removed Foreign Object", roi)

        # Fix salt & pepper noise
        if 'salt_pepper' in problems:
            roi = self.remove_salt_pepper(roi, (0, 0, w, h))
            self.add_step("Removed Salt & Pepper", roi)

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