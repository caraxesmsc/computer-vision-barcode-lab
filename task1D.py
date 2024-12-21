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
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))
        self.logger.debug(f"Added step: {description}")

    def detect_barcode_region(self, image):
        """Enhanced barcode detection with multi-scale analysis"""
        try:
            # Initial denoising for better gradient detection
            denoised = cv2.medianBlur(image, 3)
            
            # Create gradient images at multiple scales
            gradients = []
            scales = [1, 0.75, 0.5]
            
            for scale in scales:
                if scale != 1:
                    scaled = cv2.resize(denoised, None, fx=scale, fy=scale)
                else:
                    scaled = denoised.copy()
                
                # Get vertical gradient
                sobelx = cv2.Sobel(scaled, cv2.CV_64F, 1, 0, ksize=3)
                gradx = cv2.convertScaleAbs(sobelx)
                
                # Enhance vertical lines
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
                enhanced = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, kernel)
                
                gradients.append((scale, enhanced))
            
            # Find regions at each scale
            best_region = None
            best_score = 0
            
            for scale, gradient in gradients:
                # OTSU threshold for robust binarization
                thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Find connected components
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Scale back to original size
                    if scale != 1:
                        x, y = int(x/scale), int(y/scale)
                        w, h = int(w/scale), int(h/scale)
                    
                    # Validate region
                    if w < 20 or h < 10:  # Too small
                        continue
                        
                    aspect = w/float(h)
                    if not (1.5 < aspect < 5):  # Invalid aspect ratio
                        continue
                    
                    # Extract ROI
                    roi = cv2.resize(gradient[int(y*scale):int((y+h)*scale), 
                                            int(x*scale):int((x+w)*scale)], (w, h))
                    
                    # Score region
                    score = self.score_barcode_region(roi)
                    
                    if score > best_score:
                        best_score = score
                        best_region = (x, y, w, h)
            
            if best_region and best_score > 0.3:  # Minimum quality threshold
                return best_region
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error in barcode detection: {str(e)}")
            return None

    def score_barcode_region(self, roi):
        """Score potential barcode region based on vertical pattern analysis"""
        try:
            # Calculate edge density
            edge_density = np.mean(roi > 50)
            
            # Analyze vertical pattern consistency
            vertical_profile = np.mean(roi, axis=0)
            
            # Use FFT to detect regular patterns
            fft = np.abs(np.fft.fft(vertical_profile))
            pattern_score = np.max(fft[1:]) / np.mean(fft[1:])
            
            # Look for peaks in profile
            peaks, _ = find_peaks(vertical_profile, distance=3)
            peak_score = len(peaks) / roi.shape[1] if peaks.size > 0 else 0
            
            # Combine scores with weights
            total_score = (edge_density * 0.4 + 
                         min(1.0, pattern_score/100) * 0.4 +
                         peak_score * 0.2)
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error in region scoring: {str(e)}")
            return 0

    def estimate_bar_width(self, image):
        """Enhanced bar width estimation"""
        try:
            # Pre-process image
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
            
            # Get vertical edges
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            gradx = np.abs(sobelx)
            
            # Analyze middle portion of image
            middle_start = image.shape[0]//4
            middle_end = 3*image.shape[0]//4
            middle_section = gradx[middle_start:middle_end, :]
            
            # Find distances between edges
            widths = []
            for row in middle_section:
                # Normalize row
                row_norm = (row - np.min(row)) / (np.max(row) - np.min(row) + 1e-6)
                
                # Find peaks
                peaks, _ = find_peaks(row_norm, height=0.3, distance=2)
                if len(peaks) > 1:
                    # Calculate distances between adjacent peaks
                    peak_distances = np.diff(peaks)
                    
                    # Filter out outliers
                    median_distance = np.median(peak_distances)
                    valid_distances = peak_distances[
                        np.abs(peak_distances - median_distance) < median_distance]
                    
                    if len(valid_distances) > 0:
                        widths.extend(valid_distances)
            
            if not widths:
                self.logger.warning("Could not estimate bar width, using default")
                return 5
                
            # Calculate most common width
            width_counts = np.bincount(widths)
            most_common_width = np.argmax(width_counts)
            
            # Validate result
            if most_common_width < 2 or most_common_width > 20:
                most_common_width = min(max(most_common_width, 2), 20)
                
            return most_common_width
            
        except Exception as e:
            self.logger.error(f"Error estimating bar width: {str(e)}")
            return 5

    def remove_noise_and_foreign_objects(self, image):
        """Enhanced noise and foreign object removal"""
        try:
            # Estimate bar width
            bar_width = self.estimate_bar_width(image)
            self.logger.debug(f"Estimated bar width: {bar_width}")
            
            # Create multiple kernels for progressive filtering
            kernels = [
                np.ones((image.shape[0], bar_width)) / image.shape[0],
                np.ones((image.shape[0], bar_width*2)) / image.shape[0],
                np.ones((image.shape[0], bar_width*3)) / image.shape[0]
            ]
            
            # Apply progressive filtering
            stages = []
            current = image.copy()
            
            for kernel in kernels:
                filtered = cv2.filter2D(current, -1, kernel)
                stages.append(filtered)
                current = filtered
            
            # Build adaptive noise mask
            noise_mask = np.zeros_like(image, dtype=bool)
            for stage in stages:
                diff = np.abs(image - stage)
                threshold = np.mean(diff) + 2*np.std(diff)
                noise_mask |= diff > threshold
            
            # Additional median filtering for salt & pepper
            median_filtered = cv2.medianBlur(image, 3)
            median_filtered = cv2.medianBlur(median_filtered, 5)
            
            # Combine results
            result = image.copy()
            result[noise_mask] = median_filtered[noise_mask]
            
            # Final pattern-aware smoothing
            final_kernel = np.ones((image.shape[0], bar_width)) / image.shape[0]
            result = cv2.filter2D(result, -1, final_kernel)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error removing noise and foreign objects: {str(e)}")
            return image

    def fix_frequency_domain(self, image):
        """Enhanced frequency domain correction"""
        try:
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            
            rows, cols = image.shape
            crow, ccol = rows//2, cols//2
            
            # Create sophisticated filter
            x = np.linspace(-ccol, ccol-1, cols)
            y = np.linspace(-crow, crow-1, rows)
            xv, yv = np.meshgrid(x, y)
            
            # Combine multiple filter types
            r = np.sqrt(xv**2 + yv**2)
            
            # Gaussian high-pass
            sigma = 30
            gaussian = 1 - np.exp(-(r**2)/(2*sigma**2))
            
            # Butterworth high-pass
            order = 2
            cutoff = 30
            butterworth = 1 / (1 + (cutoff/r)**(2*order))
            
            # Combine filters
            mask = (gaussian + butterworth) / 2
            
            # Apply filter
            f_shift = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in frequency domain correction: {str(e)}")
            return image

    def fix_rotation(self, image):
        """Enhanced rotation correction"""
        try:
            # Add padding
            pad = int(max(image.shape) * 0.4)
            padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=255)
            
            # Find edges
            edges = cv2.Canny(padded, 50, 150)
            
            # Detect lines
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
                    
                    # Only rotate if angle is significant
                    if abs(median_angle) > 5:
                        rows, cols = padded.shape
                        
                        # Calculate new dimensions
                        angle_rad = np.abs(np.radians(median_angle))
                        new_w = int(cols*np.abs(np.cos(angle_rad)) + 
                                  rows*np.abs(np.sin(angle_rad)))
                        new_h = int(rows*np.abs(np.cos(angle_rad)) + 
                                  cols*np.abs(np.sin(angle_rad)))
                        
                        # Adjust transformation matrix
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                        M[0, 2] += (new_w - cols)/2
                        M[1, 2] += (new_h - rows)/2
                        
                        # Apply rotation
                        rotated = cv2.warpAffine(padded, M, (new_w, new_h),
                                               flags=cv2.INTER_CUBIC,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=255)
                        
                        # Find content area
                        coords = np.column_stack(np.where(rotated < 250))
                        if len(coords):
                            x_min, y_min = coords.min(axis=0)
                            x_max, y_max = coords.max(axis=0)
                            
                            # Add small padding
                            pad = 10
                            x_min = max(0, x_min - pad)
                            y_min = max(0, y_min - pad)
                            x_max = min(rotated.shape[1] - 1, x_max + pad)
                            y_max = min(rotated.shape[0] - 1, y_max + pad)
                            
                            return rotated[y_min:y_max+1, x_min:x_max+1]
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error in rotation correction: {str(e)}")
            return image

    def fix_contrast(self, image):
        """Enhanced contrast correction"""
        try:
            # Analyze histogram
            mean_val = np.mean(image)
            if mean_val < 128:  # Dark image
                # Use CLAHE with aggressive parameters
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
                
                # Additional gamma correction
                gamma = 1.3
                look_up_table = np.array([((i / 255.0) ** (1.0/gamma)) * 255
                                        for i in np.uint8([range(256)])]).astype(np.uint8)
                enhanced = cv2.LUT(enhanced, look_up_table)
            else:  # Bright image
                # Use adaptive histogram equalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
                
                # Preserve very bright areas
                bright_mask = image > 240
                enhanced[bright_mask] = image[bright_mask]
                
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error in contrast correction: {str(e)}")
            return image

    def apply_vertical_pattern_filter(self, image, bar_width=None):
        """Apply vertical pattern-based filtering"""
        try:
            if bar_width is None:
                bar_width = self.estimate_bar_width(image)
            
            # Create vertical averaging kernel
            kernel = np.ones((image.shape[0], bar_width)) / image.shape[0]
            
            # Apply filter
            filtered = cv2.filter2D(image, -1, kernel)
            
            # Create binary mask
            _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up using morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bar_width))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Error applying pattern filter: {str(e)}")
            return image

    def binarize(self, image):
        """Enhanced binarization"""
        try:
            # Normalize contrast
            p2, p98 = np.percentile(image, (2, 98))
            normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                normalized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                51,  # Block size
                7    # C constant
            )
            
            bar_width = self.estimate_bar_width(image)
            
            # Clean up using morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bar_width))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Apply vertical pattern filter
            final = self.apply_vertical_pattern_filter(cleaned, bar_width)
            
            return final
            
        except Exception as e:
            self.logger.error(f"Error in binarization: {str(e)}")
            return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    def crop_to_content(self, binary):
        """Crop image to actual barcode content"""
        try:
            # Find all non-white pixels
            coords = np.column_stack(np.where(binary < 250))
            if len(coords) == 0:
                return binary
                
            # Get bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add padding
            pad = 10
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(binary.shape[1] - 1, x_max + pad)
            y_max = min(binary.shape[0] - 1, y_max + pad)
            
            # Extract region
            cropped = binary[y_min:y_max+1, x_min:x_max+1]
            
            # Validate aspect ratio
            h, w = cropped.shape
            aspect = w / float(h)
            if 1.5 < aspect < 5:  # Valid barcode aspect ratio
                return cropped
                
            return binary
            
        except Exception as e:
            self.logger.error(f"Error cropping content: {str(e)}")
            return binary

    def final_cleanup(self, binary):
        """Final pattern-based cleanup"""
        try:
            # Get bar width
            bar_width = self.estimate_bar_width(binary)
            
            # Apply vertical pattern filter
            cleaned = self.apply_vertical_pattern_filter(binary, bar_width)
            
            # Ensure proper thresholding
            _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
            
            # Final morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bar_width))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error in final cleanup: {str(e)}")
            return binary

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        try:
            self.steps = []
            self.add_step("Original Image", image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
            
            # Fix rotation first
            rotated = self.fix_rotation(gray)
            if not np.array_equal(rotated, gray):
                self.add_step("Fixed Rotation", rotated)
                gray = rotated
            
            # Detect barcode region
            region = self.detect_barcode_region(gray)
            if region is None:
                self.logger.warning("No barcode region detected")
                return gray
                
            x, y, w, h = region
            roi = gray[y:y+h, x:x+w]
            
            # Estimate bar width for consistent processing
            bar_width = self.estimate_bar_width(roi)
            self.logger.debug(f"Detected bar width: {bar_width}")
            
            # Remove noise and foreign objects
            cleaned = self.remove_noise_and_foreign_objects(roi)
            self.add_step("Cleaned", cleaned)
            
            # Fix frequency domain issues
            freq_fixed = self.fix_frequency_domain(cleaned)
            self.add_step("Fixed Frequency", freq_fixed)
            
            # Fix contrast
            contrast_fixed = self.fix_contrast(freq_fixed)
            self.add_step("Enhanced Contrast", contrast_fixed)
            
            # Apply pattern filter
            filtered = self.apply_vertical_pattern_filter(contrast_fixed, bar_width)
            self.add_step("Pattern Filtered", filtered)
            
            # Final binarization
            binary = self.binarize(filtered)
            self.add_step("Binarized", binary)
            
            # Final cleanup
            cleaned = self.final_cleanup(binary)
            self.add_step("Final Cleanup", cleaned)
            
            # Crop to content
            cropped = self.crop_to_content(cleaned)
            self.add_step("Final Result", cropped)
            
            return cropped
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return image

    def visualize_steps(self):
        """Visualize processing steps"""
        try:
            n_steps = len(self.steps)
            cols = min(4, n_steps)
            rows = (n_steps + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            
            if rows * cols == 1:
                axes = np.array([axes])
            
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
            
        except Exception as e:
            self.logger.error(f"Error visualizing steps: {str(e)}")

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