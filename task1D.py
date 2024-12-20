import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.debug = False  # For additional debug visualization

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def estimate_bar_thickness(self, image):
        """Estimate barcode bar thickness for pattern-aware processing"""
        # Initial preprocessing
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
        
        min_width = max(1, min(int(np.percentile(widths, 10)), 10))  # Cap at 10
        pattern_width = max(3, min(int(np.median(widths)), 30))  # Cap at 30
        
        return min_width, pattern_width

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
            
        # Filter contours by aspect ratio and density
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
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def detect_image_problems(self, image):
        """Optimized problem detection combining best approaches"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Find barcode region
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])
            
        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]
        
        # Estimate bar pattern info first - needed for multiple detections
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)
        
        # Get vertical and horizontal profiles for multiple detections
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        
        # 1. Blur detection (combined approach)
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.mean(edges > 0)
        if laplacian_var < 90 and edge_density < 0.15:
            problems.append('blur')
            
        # 2. Contrast detection (enhanced)
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')
            
        # 3. Salt and pepper detection
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.percentile(noise_diff, 95) > 40:
            problems.append('salt_pepper')
            
        # 4. Compression artifacts
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        blocks = cv2.boxFilter(magnitude, -1, (8,8))
        if np.std(blocks) > 15:
            problems.append('compression_artifacts')
            
        # 5. Frequency domain detection
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')
            
        # 6. Foreign object detection (enhanced)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        v_threshold = np.mean(vertical_profile) + 2.5 * v_std
        h_threshold = np.mean(horizontal_profile) + 2.5 * h_std
        
        v_anomalies = np.abs(vertical_profile - np.mean(vertical_profile)) > v_threshold
        h_anomalies = np.abs(horizontal_profile - np.mean(horizontal_profile)) > h_threshold
        
        if np.sum(v_anomalies) > min_bar_width * 2 or np.sum(h_anomalies) > min_bar_width * 2:
            problems.append('foreign_object')
            
        # 7. Rotation detection (from best performing version)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')
                
        return problems, barcode_region

    def remove_foreign_object(self, image, barcode_region):
        """Enhanced foreign object removal using vertical kernel filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Get bar pattern info
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)
        
        # Create vertical averaging kernel
        kernel = np.ones((h, min_bar_width)) / h
        
        # Get initial averaged profile
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        
        # Calculate thresholds
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        
        cleaned = roi.copy()
        
        # Process each column
        for i in range(0, w - min_bar_width + 1):
            region = roi[:, i:i+min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            
            # Check if column needs cleaning
            if np.any(np.abs(vertical_avg - np.mean(vertical_avg)) > 2.5 * v_std):
                # Get neighboring columns for interpolation
                left_cols = []
                right_cols = []
                
                # Look for good columns within 2*pattern_width
                for offset in range(-2*pattern_width, -min_bar_width):
                    if i + offset >= 0:
                        check_col = roi[:, i+offset]
                        if np.std(check_col) < v_std * 2:
                            left_cols.append(check_col)
                            
                for offset in range(min_bar_width, 2*pattern_width):
                    if i + offset < w:
                        check_col = roi[:, i+offset]
                        if np.std(check_col) < v_std * 2:
                            right_cols.append(check_col)
                            
                # Interpolate from good neighbors
                if left_cols and right_cols:
                    avg_col = np.mean([np.mean(left_cols, axis=0), 
                                     np.mean(right_cols, axis=0)], axis=0)
                    cleaned[:, i:i+min_bar_width] = \
                        np.tile(avg_col, (min_bar_width, 1)).T
                
        # Apply vertical filtering
        final = cv2.filter2D(cleaned, -1, kernel)
        
        result = image.copy()
        result[y:y+h, x:x+w] = final
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Fix frequency domain issues using Gaussian high-pass filter"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2
        
        # Create Gaussian high-pass filter
        x = np.linspace(-ccol, ccol-1, cols)
        y = np.linspace(-crow, crow-1, rows)
        xv, yv = np.meshgrid(x, y)
        sigma = 30
        gaussian = np.exp(-(xv**2 + yv**2)/(2*sigma**2))
        mask = 1 - gaussian
        
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_blur(self, image, barcode_region):
        """Two-stage deblurring with careful parameters"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # First stage: Unsharp masking with moderate parameters
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.3, gaussian, -0.3, 0)
        
        # Check if more sharpening needed
        lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if lap_var < 80:
            # Mild second stage
            gaussian2 = cv2.GaussianBlur(unsharp, (0, 0), 1)
            unsharp = cv2.addWeighted(unsharp, 1.2, gaussian2, -0.2, 0)
        
        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def fix_contrast(self, image, barcode_region):
        """Enhanced contrast correction with preservation of extremes"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # CLAHE with moderate parameters
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        
        # Preserve pure black and white
        enhanced[roi < 30] = 0
        enhanced[roi > 225] = 255
        
        # Adaptive gamma correction
        mean_intensity = np.mean(enhanced)
        gamma = 1.1 if mean_intensity < 128 else 0.9
        inv_gamma = 1.0/gamma
        table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)
        
        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def fix_rotation(self, image, barcode_region):
        """Rotation correction with proper padding and preservation"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Add padding before rotation
        pad = int(max(w, h) * 0.3)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, 
                                   cv2.BORDER_CONSTANT, value=255)
        
        edges = cv2.Canny(padded, 50, 150)
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
                rows, cols = padded.shape
                
                # Calculate new dimensions
                angle_rad = abs(np.radians(median_angle))
                new_width = int(cols * abs(np.cos(angle_rad)) + rows * abs(np.sin(angle_rad)))
                new_height = int(rows * abs(np.cos(angle_rad)) + cols * abs(np.sin(angle_rad)))
                
                # Adjust rotation matrix
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                matrix[0, 2] += (new_width - cols) / 2
                matrix[1, 2] += (new_height - rows) / 2
                
                # Perform rotation
                rotated = cv2.warpAffine(padded, matrix, (new_width, new_height),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
                
                # Find barcode area in rotated image
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

    def remove_salt_pepper(self, image, barcode_region):
        """Enhanced salt and pepper removal with progressive filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Multiple passes of median filtering
        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)
        
        # Check noise level for additional filtering
        noise_level = np.std(filtered - roi)
        if noise_level > 30:
            filtered = cv2.medianBlur(filtered, 3)
        
        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def fix_compression_artifacts(self, image, barcode_region):
        """Fix compression artifacts using non-local means denoising"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        denoised = cv2.fastNlMeansDenoising(
            roi,
            None,
            h=5,  # Lower strength to preserve detail
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        result = image.copy()
        result[y:y+h, x:x+w] = denoised
        return result

    def binarize(self, image):
        """Improved binarization with hole prevention"""
        # Normalize contrast first
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        # Adaptive threshold with carefully chosen parameters
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Window size
            7    # Adjusted offset
        )
        
        # Light cleanup using morphology
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def final_barcode_cleanup(self, binary):
        """Final cleanup using bar pattern-aware vertical filtering"""
        min_bar_width, _ = self.estimate_bar_thickness(binary)
        min_bar_width = max(1, min(min_bar_width, 10))  # Ensure reasonable size
        
        # Create vertical averaging kernel
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]
        
        # Apply kernel along horizontal axis
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
        
        # Detect problems and barcode region
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)
        
        if barcode_region is None:
            return gray
            
        processed = gray.copy()
        x, y, w, h = barcode_region
        
        # Fix rotation first if needed
        if 'rotation' in problems:
            processed = self.fix_rotation(processed, barcode_region)
            self.add_step("Fixed Rotation", processed)
            
            # Update barcode region after rotation
            new_region = self.detect_barcode_region(processed)
            if new_region:
                barcode_region = new_region
                x, y, w, h = barcode_region
        
        # Process the ROI
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed, barcode_region)
            self.add_step("Removed Foreign Object", processed)
        
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed, barcode_region)
            self.add_step("Fixed Frequency Domain", processed)
        
        if 'compression_artifacts' in problems:
            processed = self.fix_compression_artifacts(processed, barcode_region)
            self.add_step("Fixed Compression Artifacts", processed)
        
        if 'salt_pepper' in problems:
            processed = self.remove_salt_pepper(processed, barcode_region)
            self.add_step("Removed Salt & Pepper", processed)
        
        if 'blur' in problems:
            processed = self.fix_blur(processed, barcode_region)
            self.add_step("Fixed Blur", processed)
        
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed, barcode_region)
            self.add_step("Enhanced Contrast", processed)
        
        # Extract ROI for final processing
        roi = processed[y:y+h, x:x+w]
        
        # Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)
        
        # Final cleanup using vertical averaging
        cleaned = self.final_barcode_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)
        
        return cleaned

    def visualize_steps(self):
        """Enhanced visualization with proper subplot handling"""
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