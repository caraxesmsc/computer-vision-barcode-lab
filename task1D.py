import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks
import logging
import datetime
import os

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/barcode_processor_{timestamp}.log'

        self.logger = logging.getLogger('BarcodeProcessor')
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))
        self.logger.debug(f"Added step: {description}")

    def detect_barcode_region(self, image):
        """Find barcode region using edge density"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        # Find regions with high concentration of vertical lines
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find connected components
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by aspect ratio and area
        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 10:
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        return max(valid_regions, 
                  key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def calculate_quality_score(self, image, region=None):
        """Calculate quality score based on multiple metrics"""
        if region is None:
            roi = image
        else:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]

        # Vertical line consistency (barcode characteristic)
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        vertical_score = np.mean(np.abs(sobelx))

        # Contrast
        p5, p95 = np.percentile(roi, [5, 95])
        contrast_score = p95 - p5

        # Edge clarity
        edges = cv2.Canny(roi, 50, 150)
        edge_score = np.mean(edges)

        # Noise level (inverse)
        noise_score = 1.0 / (cv2.Laplacian(roi, cv2.CV_64F).var() + 1e-10)

        # Combine scores with weights
        total_score = (vertical_score * 0.4 + 
                      contrast_score * 0.3 + 
                      edge_score * 0.2 + 
                      noise_score * 0.1)

        self.logger.debug(f"Quality scores - Vertical: {vertical_score:.2f}, "
                         f"Contrast: {contrast_score:.2f}, Edge: {edge_score:.2f}")

        return total_score

    def compare_results(self, img1, img2, current_stage):
        """Compare two results and choose the better one"""
        self.logger.info(f"Comparing results from {current_stage}")

        # Try to detect barcode in both
        region1 = self.detect_barcode_region(img1)
        region2 = self.detect_barcode_region(img2)

        # If one has barcode region and other doesn't
        if region1 is None and region2 is not None:
            self.logger.info(f"{current_stage}: Choosing new result (found barcode region)")
            return img2, region2
        if region2 is None and region1 is not None:
            self.logger.info(f"{current_stage}: Keeping previous result (found barcode region)")
            return img1, region1

        # If both have regions, compare quality
        if region1 and region2:
            score1 = self.calculate_quality_score(img1, region1)
            score2 = self.calculate_quality_score(img2, region2)
            
            self.logger.debug(f"Quality scores - Previous: {score1:.2f}, New: {score2:.2f}")
            
            if score2 > score1 * 1.1:  # Require 10% improvement to switch
                self.logger.info(f"{current_stage}: Choosing new result (better quality)")
                return img2, region2
            else:
                self.logger.info(f"{current_stage}: Keeping previous result")
                return img1, region1

        # If no regions found, compare whole images
        score1 = self.calculate_quality_score(img1)
        score2 = self.calculate_quality_score(img2)
        if score2 > score1 * 1.1:
            self.logger.info(f"{current_stage}: Choosing new result (better overall quality)")
            return img2, None
        else:
            self.logger.info(f"{current_stage}: Keeping previous result")
            return img1, None

    def initial_cleanup(self, image):
        """Basic image enhancement - First stage of pipeline"""
        self.logger.info("Performing initial cleanup")
        
        # Initial denoising
        denoised = cv2.medianBlur(image, 3)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

        # OTSU thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def check_rotation_angle(self, image):
        """Check if image needs rotation"""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(image.shape)//3)
        
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
                if abs(median_angle) > 10:  # More conservative threshold
                    return median_angle
        
        return 0

    def fix_rotation(self, image):
        """Fix rotation with proper padding"""
        angle = self.check_rotation_angle(image)
        
        if abs(angle) <= 10:  # Don't rotate if angle is small
            return image

        # Add padding before rotation
        pad = int(max(image.shape) * 0.3)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=255)

        rows, cols = padded.shape
        matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

        # Calculate new dimensions
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((rows * sin) + (cols * cos))
        new_h = int((rows * cos) + (cols * sin))

        # Adjust transformation matrix
        matrix[0, 2] += (new_w/2) - cols/2
        matrix[1, 2] += (new_h/2) - rows/2

        # Apply rotation with border replication
        rotated = cv2.warpAffine(padded, matrix, (new_w, new_h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)

        # Find content bbox to crop padding
        coords = np.column_stack(np.where(rotated < 250))
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        return rotated[y_min:y_max+1, x_min:x_max+1]

    # Additional methods will be in part 2...
    def remove_foreign_object(self, image):
        """Remove foreign objects using vertical pattern analysis"""
        # Estimate bar width
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=20, maxLineGap=10)
            
        bar_width = 5  # Default
        if lines is not None:
            widths = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10:  # Nearly vertical lines
                    widths.append(abs(x2 - x1))
            if widths:
                bar_width = max(3, int(np.median(widths)))

        # Create multiple kernels
        kernel1 = np.ones((image.shape[0], bar_width)) / image.shape[0]
        kernel2 = np.ones((image.shape[0], bar_width * 3)) / image.shape[0]
        kernel3 = np.ones((image.shape[0], 1000)) / image.shape[0]  # Very tall kernel
        
        # Progressive filtering
        filtered1 = cv2.filter2D(image, -1, kernel1)
        filtered2 = cv2.filter2D(filtered1, -1, kernel2)
        filtered3 = cv2.filter2D(filtered2, -1, kernel3)
        
        # Final thresholding
        _, result = cv2.threshold(filtered3, 127, 255, cv2.THRESH_BINARY)
        
        return result

    def fix_frequency_domain(self, image):
        """Fix frequency domain issues using high-pass filter"""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        
        # Create high-pass filter
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
        mask[mask_area] = 0
        
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def fix_contrast(self, image):
        """Fix contrast based on image statistics"""
        mean_val = np.mean(image)
        if mean_val < 128:  # Dark image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)
        else:  # Bright image
            p2, p98 = np.percentile(image, (2, 98))
            return np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)

    def final_cleanup(self, image, bar_width=None):
        """Final cleanup using vertical pattern averaging"""
        if bar_width is None:
            edges = cv2.Canny(image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                                   minLineLength=20, maxLineGap=10)
            bar_width = 5
            if lines is not None:
                widths = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) < 10:
                        widths.append(abs(x2 - x1))
                if widths:
                    bar_width = max(3, int(np.median(widths)))

        kernel = np.ones((image.shape[0], bar_width)) / image.shape[0]
        cleaned = cv2.filter2D(image, -1, kernel)
        _, binary = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        
        return binary

    def preprocess_image(self, image):
        """Main preprocessing pipeline with result comparison"""
        self.logger.info("Starting preprocessing pipeline")
        self.steps = []
        self.add_step("Original Image", image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Stage 1: Initial Enhancement
        enhanced = self.initial_cleanup(gray)
        self.add_step("Initial Enhancement", enhanced)
        
        # Save the good result
        best_result = enhanced.copy()
        best_region = self.detect_barcode_region(best_result)

        # Stage 2: Check Rotation
        angle = self.check_rotation_angle(enhanced)
        if abs(angle) > 10:  # Only rotate if significant angle
            rotated = self.fix_rotation(enhanced)
            rotated_region = self.detect_barcode_region(rotated)
            if rotated_region:  # Only use rotated version if barcode detected
                best_result, best_region = self.compare_results(best_result, rotated, "Rotation")
                if best_result is rotated:
                    self.add_step("Fixed Rotation", best_result)

        # Stage 3: Try Foreign Object Removal
        if best_region:  # Only if we have a valid barcode region
            x, y, w, h = best_region
            roi = best_result[y:y+h, x:x+w]
            
            # Try foreign object removal
            cleaned = self.remove_foreign_object(roi)
            full_cleaned = best_result.copy()
            full_cleaned[y:y+h, x:x+w] = cleaned
            
            # Compare and update if better
            best_result, best_region = self.compare_results(best_result, full_cleaned, "Foreign Object")
            if best_result is full_cleaned:
                self.add_step("Removed Foreign Object", best_result)

        # Stage 4: Frequency Domain Fix
        if best_region:
            x, y, w, h = best_region
            roi = best_result[y:y+h, x:x+w]
            
            freq_fixed = self.fix_frequency_domain(roi)
            full_freq = best_result.copy()
            full_freq[y:y+h, x:x+w] = freq_fixed
            
            best_result, best_region = self.compare_results(best_result, full_freq, "Frequency")
            if best_result is full_freq:
                self.add_step("Fixed Frequency Domain", best_result)

        # Stage 5: Contrast Enhancement
        if best_region:
            x, y, w, h = best_region
            roi = best_result[y:y+h, x:x+w]
            
            contrast_fixed = self.fix_contrast(roi)
            full_contrast = best_result.copy()
            full_contrast[y:y+h, x:x+w] = contrast_fixed
            
            best_result, best_region = self.compare_results(best_result, full_contrast, "Contrast")
            if best_result is full_contrast:
                self.add_step("Enhanced Contrast", best_result)

        # Final Stage: Cleanup and Crop
        if best_region:
            x, y, w, h = best_region
            roi = best_result[y:y+h, x:x+w]
            
            # Only apply final cleanup if we have a good barcode region
            final_cleaned = self.final_cleanup(roi)
            
            # Compare one last time
            full_final = best_result.copy()
            full_final[y:y+h, x:x+w] = final_cleaned
            
            # For final result, be more conservative about changes
            score_orig = self.calculate_quality_score(best_result, best_region)
            score_final = self.calculate_quality_score(full_final, best_region)
            
            if score_final > score_orig * 1.2:  # Require 20% improvement for final change
                best_result = full_final
                self.add_step("Final Cleanup", best_result)
                
            # Crop to just the barcode region with padding
            pad_x = int(0.1 * w)
            pad_y = int(0.1 * h)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(best_result.shape[1] - x, w + 2*pad_x)
            h = min(best_result.shape[0] - y, h + 2*pad_y)
            
            final_result = best_result[y:y+h, x:x+w]
            self.add_step("Final Result", final_result)
            return final_result

        # If no barcode region found, return the best result we have
        return best_result

    def visualize_steps(self):
        """Visualize all processing steps"""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
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