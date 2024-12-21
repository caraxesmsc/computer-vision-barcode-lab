import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks
import logging
import datetime
import os
from enum import Enum, auto

class ImageQualityMetric(Enum):
    CONTRAST = auto()
    SHARPNESS = auto()
    NOISE = auto()
    PATTERN_CONSISTENCY = auto()

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.setup_logging()
        self.min_quality_threshold = 50
        self.improvement_threshold = 1.1  # 10% improvement required

    def setup_logging(self):
        """Setup detailed logging configuration"""
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

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def add_step(self, description, image):
        """Save preprocessing step with quality metrics"""
        self.steps.append((description, image.copy()))
        quality = self.calculate_quality_metrics(image)
        self.logger.debug(f"Step: {description} - Quality Metrics: {quality}")

    def calculate_quality_metrics(self, image):
        """Calculate comprehensive quality metrics for an image"""
        try:
            metrics = {}
            
            # Contrast
            p5, p95 = np.percentile(image, [5, 95])
            metrics[ImageQualityMetric.CONTRAST] = (p95 - p5) / 255.0
            
            # Sharpness
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            metrics[ImageQualityMetric.SHARPNESS] = min(1.0, laplacian_var / 500.0)
            
            # Noise
            median = cv2.medianBlur(image, 3)
            noise = np.mean(np.abs(image - median))
            metrics[ImageQualityMetric.NOISE] = 1.0 - min(1.0, noise / 50.0)
            
            # Pattern Consistency (for barcodes)
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            vertical_consistency = np.std(np.mean(np.abs(sobelx), axis=0))
            metrics[ImageQualityMetric.PATTERN_CONSISTENCY] = min(1.0, vertical_consistency / 50.0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {str(e)}")
            return {}

    def is_improvement(self, original, processed, metric_weights=None):
        """Check if processed image is an improvement over original"""
        if metric_weights is None:
            metric_weights = {
                ImageQualityMetric.CONTRAST: 0.3,
                ImageQualityMetric.SHARPNESS: 0.3,
                ImageQualityMetric.NOISE: 0.2,
                ImageQualityMetric.PATTERN_CONSISTENCY: 0.2
            }
            
        orig_metrics = self.calculate_quality_metrics(original)
        proc_metrics = self.calculate_quality_metrics(processed)
        
        if not orig_metrics or not proc_metrics:
            return False
            
        orig_score = sum(metric_weights[m] * orig_metrics[m] for m in metric_weights)
        proc_score = sum(metric_weights[m] * proc_metrics[m] for m in metric_weights)
        
        return proc_score > orig_score * self.improvement_threshold

    def estimate_bar_width(self, image):
        """Estimate barcode bar width using vertical gradients and peak detection"""
        try:
            # Enhance vertical edges
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gradx = np.abs(sobelx)
            
            # Analyze middle rows for more reliable estimation
            middle_rows = gradx[image.shape[0]//4:3*image.shape[0]//4, :]
            
            # Find peaks in each row
            widths = []
            for row in middle_rows:
                # Normalize row
                row_norm = (row - np.min(row)) / (np.max(row) - np.min(row))
                peaks, _ = find_peaks(row_norm, height=0.3, distance=2)
                if len(peaks) > 1:
                    # Calculate distances between adjacent peaks
                    peak_distances = np.diff(peaks)
                    # Filter out outliers
                    median_distance = np.median(peak_distances)
                    valid_distances = peak_distances[np.abs(peak_distances - median_distance) < median_distance]
                    if len(valid_distances) > 0:
                        widths.extend(valid_distances)
            
            if not widths:
                self.logger.warning("Could not estimate bar width, using default")
                return 5
                
            # Use the most common width
            width_counts = np.bincount(widths)
            most_common_width = np.argmax(width_counts)
            
            # Validate the result
            if most_common_width < 2 or most_common_width > 20:
                self.logger.warning(f"Unusual bar width detected: {most_common_width}, adjusting")
                most_common_width = min(max(most_common_width, 2), 20)
                
            return most_common_width
            
        except Exception as e:
            self.logger.error(f"Error estimating bar width: {str(e)}")
            return 5

    def detect_barcode_region(self, image):
        """Enhanced barcode region detection"""
        try:
            # Create vertical gradient image
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gradx = cv2.convertScaleAbs(sobelx)
            
            # Enhance vertical lines
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, morph_kernel)
            
            # Threshold to get potential regions
            blur = cv2.GaussianBlur(gradx, (21, 21), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find connected components
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.warning("No potential barcode regions found")
                return None
                
            # Score regions based on multiple criteria
            valid_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 20 or h < 10:  # Too small
                    continue
                    
                aspect_ratio = w / float(h)
                if not (1.5 < aspect_ratio < 5):  # Invalid aspect ratio
                    continue
                    
                roi = gradx[y:y+h, x:x+w]
                
                # Calculate various metrics
                edge_density = np.sum(roi > 50) / (w * h)
                vertical_consistency = np.std(np.mean(roi, axis=0))
                area_score = (w * h) / (image.shape[0] * image.shape[1])
                
                # Combine metrics into final score
                score = (edge_density * 0.4 + 
                        (1 - vertical_consistency/100) * 0.3 + 
                        area_score * 0.3)
                
                if edge_density > 0.1:  # Minimum requirement
                    valid_regions.append((score, (x, y, w, h)))
                    
            if not valid_regions:
                self.logger.warning("No valid barcode regions found")
                return None
                
            # Return region with highest score
            best_region = max(valid_regions, key=lambda x: x[0])[1]
            self.logger.info(f"Found barcode region: {best_region}")
            return best_region
            
        except Exception as e:
            self.logger.error(f"Error detecting barcode region: {str(e)}")
            return None

    def remove_foreign_object(self, image):
        """Enhanced foreign object removal using bar pattern analysis"""
        try:
            # Estimate bar width for kernel size
            bar_width = self.estimate_bar_width(image)
            self.logger.debug(f"Estimated bar width: {bar_width}")
            
            # Create vertical averaging kernels of different sizes
            kernel1 = np.ones((image.shape[0], bar_width)) / image.shape[0]
            kernel2 = np.ones((image.shape[0], bar_width * 2)) / image.shape[0]
            kernel3 = np.ones((image.shape[0], bar_width * 3)) / image.shape[0]
            
            # Apply kernels progressively
            filtered1 = cv2.filter2D(image, -1, kernel1)
            filtered2 = cv2.filter2D(filtered1, -1, kernel2)
            filtered3 = cv2.filter2D(filtered2, -1, kernel3)
            
            # Analyze vertical profiles for anomalies
            profile = np.mean(image, axis=0)
            filtered_profile = np.mean(filtered3, axis=0)
            
            # Calculate local thresholds
            window = bar_width * 5
            thresholds = []
            for i in range(len(profile)):
                start = max(0, i - window)
                end = min(len(profile), i + window)
                local_std = np.std(profile[start:end])
                thresholds.append(local_std * 2.5)
            
            # Find anomalous regions
            anomalies = np.abs(profile - filtered_profile) > thresholds
            
            # Create mask for blending
            mask = np.zeros_like(image, dtype=float)
            for i in range(len(anomalies)):
                if anomalies[i]:
                    mask[:, i] = 1.0
            
            # Smooth mask edges
            mask = cv2.GaussianBlur(mask, (bar_width*2+1, bar_width*2+1), 0)
            
            # Blend original and filtered images
            result = (1 - mask) * image + mask * filtered3
            
            return result.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error removing foreign object: {str(e)}")
            return image

    def fix_frequency_domain(self, image):
        """Enhanced frequency domain correction"""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            
            rows, cols = image.shape
            crow, ccol = rows//2, cols//2
            
            # Create more sophisticated filter
            x = np.linspace(-ccol, ccol-1, cols)
            y = np.linspace(-crow, crow-1, rows)
            xv, yv = np.meshgrid(x, y)
            
            # Combine Gaussian and Butterworth filters
            r = np.sqrt(xv**2 + yv**2)
            butterworth_order = 2
            cutoff_freq = 30
            
            gaussian = 1 - np.exp(-(r**2)/(2*cutoff_freq**2))
            butterworth = 1 / (1 + (r/cutoff_freq)**(2*butterworth_order))
            
            # Combine filters
            mask = (gaussian + butterworth) / 2
            
            # Apply filter
            f_shift = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            # Normalize result
            result = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error fixing frequency domain: {str(e)}")
            return image

    def remove_salt_pepper(self, image):
        """Enhanced salt and pepper noise removal"""
        try:
            bar_width = self.estimate_bar_width(image)
            
            # Progressive median filtering
            filtered = image.copy()
            kernel_sizes = [3, 5]
            
            for size in kernel_sizes:
                filtered_temp = cv2.medianBlur(filtered, size)
                
                # Calculate noise mask
                diff = np.abs(filtered - filtered_temp)
                noise_mask = diff > np.mean(diff) + 2*np.std(diff)
                
                # Only update noisy pixels
                filtered[noise_mask] = filtered_temp[noise_mask]
            
            # Final vertical averaging based on bar width
            kernel = np.ones((image.shape[0], bar_width)) / image.shape[0]
            final = cv2.filter2D(filtered, -1, kernel)
            
            # Blend based on noise levels
            noise_levels = np.abs(image - final)
            blend_mask = noise_levels > np.mean(noise_levels) + np.std(noise_levels)
            result = np.where(blend_mask, final, image)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error removing salt and pepper noise: {str(e)}")
            return image

    def fix_blur(self, image):
        """Enhanced deblurring with edge preservation"""
        try:
            # First pass: Unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 2)
            unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            # Second pass: Edge enhancement
            kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
            
            # Check if more sharpening needed
            lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
            if lap_var < 150:
                # Apply edge-preserving filter before sharpening
                bilateral = cv2.bilateralFilter(unsharp, 9, 75, 75)
                sharpened = cv2.filter2D(bilateral, -1, kernel)
                result = sharpened
            else:
                result = unsharp
            
            return result.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error fixing blur: {str(e)}")
            return image

    def fix_contrast(self, image):
        """Enhanced contrast adjustment with histogram analysis"""
        try:
            # Analyze histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            cumsum = np.cumsum(hist_norm)
            
            # Calculate statistics
            mean_val = np.mean(image)
            std_val = np.std(image)
            
            if mean_val < 128:  # Dark image
                # Use CLAHE for dark images
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
            self.logger.error(f"Error fixing contrast: {str(e)}")
            return image

    def fix_rotation(self, image):
        """Enhanced rotation correction with improved padding"""
        try:
            # Add large padding to prevent info loss
            pad = int(max(image.shape) * 0.4)  # 40% padding
            padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=255)
            
            # Detect edges
            edges = cv2.Canny(padded, 50, 150)
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
                    rows, cols = padded.shape
                    center = (cols//2, rows//2)
                    
                    # Calculate new dimensions post-rotation
                    angle_rad = np.abs(np.radians(median_angle))
                    new_width = int(cols * np.abs(np.cos(angle_rad)) + rows * np.abs(np.sin(angle_rad)))
                    new_height = int(rows * np.abs(np.cos(angle_rad)) + cols * np.abs(np.sin(angle_rad)))
                    
                    # Adjust rotation matrix
                    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    matrix[0, 2] += (new_width - cols) / 2
                    matrix[1, 2] += (new_height - rows) / 2
                    
                    # Apply rotation
                    rotated = cv2.warpAffine(padded, matrix, (new_width, new_height),
                                           flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=255)
                    
                    # Find content area
                    coords = np.column_stack(np.where(rotated < 250))
                    if len(coords) > 0:
                        x_min, y_min = coords.min(axis=0)
                        x_max, y_max = coords.max(axis=0)
                        
                        # Add small padding to cropped area
                        pad = 10
                        x_min = max(0, x_min - pad)
                        y_min = max(0, y_min - pad)
                        x_max = min(rotated.shape[1] - 1, x_max + pad)
                        y_max = min(rotated.shape[0] - 1, y_max + pad)
                        
                        return rotated[y_min:y_max+1, x_min:x_max+1]
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error fixing rotation: {str(e)}")
            return image

    def final_cleanup(self, binary):
        """Final cleanup using bar pattern analysis"""
        try:
            # Estimate bar width
            bar_width = self.estimate_bar_width(binary)
            
            # Create vertical averaging kernel
            kernel = np.ones((binary.shape[0], bar_width)) / binary.shape[0]
            
            # Apply kernel progressively
            cleaned = np.zeros_like(binary)
            for i in range(0, binary.shape[1] - bar_width + 1):
                section = binary[:, i:i+bar_width]
                avg = np.mean(section, axis=1)
                threshold = np.mean(avg)
                cleaned[:, i] = (avg > threshold) * 255
                
            # Clean up edges
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error in final cleanup: {str(e)}")
            return binary

    def preprocess_image(self, image):
        """Main preprocessing pipeline with quality checks"""
        try:
            self.steps = []
            self.add_step("Original Image", image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
            
            # Initial problem detection
            self.logger.info("Starting preprocessing pipeline")
            orig_quality = self.calculate_quality_metrics(gray)
            
            # Fix rotation first if needed
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta)
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                if angles and abs(np.median(angles)) > 5:
                    gray_rotated = self.fix_rotation(gray)
                    if self.is_improvement(gray, gray_rotated):
                        gray = gray_rotated
                        self.add_step("Fixed Rotation", gray)
            
            # Detect barcode region
            region = self.detect_barcode_region(gray)
            if region is None:
                self.logger.warning("No barcode region detected")
                return gray
                
            x, y, w, h = region
            roi = gray[y:y+h, x:x+w]
            
            # Process ROI
            processed = roi.copy()
            
            # Apply fixes with quality checks
            fixes = [
                ('foreign_object', self.remove_foreign_object),
                ('frequency_domain', self.fix_frequency_domain),
                ('salt_pepper', self.remove_salt_pepper),
                ('blur', self.fix_blur),
                ('contrast', self.fix_contrast)
            ]
            
            for problem, fix_func in fixes:
                fixed = fix_func(processed)
                if self.is_improvement(processed, fixed):
                    processed = fixed
                    self.add_step(f"Fixed {problem}", processed)
            
            # Final binarization
            binary = self.binarize(processed)
            self.add_step("Binarized", binary)
            
            # Final cleanup
            cleaned = self.final_cleanup(binary)
            self.add_step("Final Cleanup", cleaned)
            
            # Calculate final quality improvement
            final_quality = self.calculate_quality_metrics(cleaned)
            quality_improvement = sum(final_quality.values()) / sum(orig_quality.values())
            self.logger.info(f"Overall quality improvement: {quality_improvement:.2f}x")
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return image

    def binarize(self, image):
        """Enhanced binarization with local thresholding"""
        try:
            # Normalize contrast first
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
            
            # Clean up noise with morphological operations
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Error in binarization: {str(e)}")
            return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    def visualize_steps(self):
        """Visualize preprocessing steps with quality metrics"""
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
                    
                # Add quality metrics to title
                metrics = self.calculate_quality_metrics(img)
                quality_str = f"\nQ:{sum(metrics.values())/len(metrics):.2f}"
                ax.set_title(description + quality_str, fontsize=8)
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