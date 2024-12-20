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
        
        # Configure detection thresholds
        self.THRESHOLDS = {
            'blur': {'laplacian_var': 80, 'edge_density': 0.12},
            'contrast': {'range_min': 70},
            'salt_pepper': {'diff_threshold': 40},
            'compression': {'block_var': 30},
            'frequency': {'std_threshold': 45},
            'rotation': {'angle_threshold': 5}
        }

    def setup_logging(self):
        """Setup logging configuration"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/barcode_processor_{timestamp}.log'

        self.logger = logging.getLogger('BarcodeProcessor')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with detailed logging
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler with less verbose logging
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def add_step(self, description, image):
        """Save each step for visualization with logging"""
        self.steps.append((description, image.copy()))
        mean_val = np.mean(image) if image is not None else 0
        std_val = np.std(image) if image is not None else 0
        self.logger.debug(f"Step: {description} - Mean: {mean_val:.2f}, Std: {std_val:.2f}")

    def detect_foreign_object(self, image):
        """Detect foreign objects using vertical profile analysis"""
        # Calculate vertical and horizontal profiles
        vertical_profile = np.mean(image, axis=0)
        horizontal_profile = np.mean(image, axis=1)
        
        # Calculate standard deviations
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        
        # Look for anomalous regions
        v_threshold = np.mean(vertical_profile) + 2.5 * v_std
        h_threshold = np.mean(horizontal_profile) + 2.5 * h_std
        
        v_anomalies = np.abs(vertical_profile - np.mean(vertical_profile)) > v_threshold
        h_anomalies = np.abs(horizontal_profile - np.mean(horizontal_profile)) > h_threshold
        
        # Log findings
        self.logger.debug(f"Foreign object detection - V_anomalies: {np.sum(v_anomalies)}, H_anomalies: {np.sum(h_anomalies)}")
        
        # Return True if significant anomalies found
        return np.sum(v_anomalies) > 5 or np.sum(h_anomalies) > 5
        
    def detect_blur(self, image):
        """Improved blur detection"""
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.mean(edges > 0)
        
        self.logger.debug(f"Blur detection - Laplacian var: {laplacian_var:.2f}, Edge density: {edge_density:.2f}")
        return laplacian_var < self.THRESHOLDS['blur']['laplacian_var']

    def detect_low_contrast(self, image):
        """Improved contrast detection"""
        p5, p95 = np.percentile(image, [5, 95])
        contrast_range = p95 - p5
        self.logger.debug(f"Contrast range: {contrast_range:.2f}")
        return contrast_range < self.THRESHOLDS['contrast']['range_min']

    def detect_salt_pepper(self, image):
        """Enhanced salt and pepper detection"""
        median = cv2.medianBlur(image, 3)
        noise_diff = np.abs(image - median)
        noise_level = np.percentile(noise_diff, 95)
        self.logger.debug(f"Salt & pepper noise level: {noise_level:.2f}")
        return noise_level > self.THRESHOLDS['salt_pepper']['diff_threshold']

    def detect_frequency_domain_issues(self, image):
        """Improved frequency domain analysis"""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        spectrum_std = np.std(magnitude_spectrum)
        self.logger.debug(f"Frequency domain std: {spectrum_std:.2f}")
        return spectrum_std > self.THRESHOLDS['frequency']['std_threshold']

    def detect_rotation(self, image):
        """Enhanced rotation detection"""
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
                self.logger.debug(f"Detected rotation angle: {median_angle:.2f}")
                return abs(median_angle) > self.THRESHOLDS['rotation']['angle_threshold']
        return False

    def fix_salt_and_pepper(self, image):
        """Enhanced salt and pepper removal"""
        self.logger.info("Removing salt & pepper noise")
        # Progressive median filtering
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        
        # Check if more filtering needed
        noise_level = np.std(result - image)
        if noise_level > 30:
            result = cv2.medianBlur(result, 3)
        
        return result

    def fix_blur(self, image):
        """Two-stage blur correction"""
        self.logger.info("Fixing blur")
        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 1.7, gaussian, -0.7, 0)
        
        # Second stage: Edge enhancement
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(unsharp, -1, kernel)
        
        return sharpened

    def fix_contrast(self, image):
        """Enhanced contrast correction"""
        self.logger.info("Enhancing contrast")
        # CLAHE with careful parameters
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Preserve black and white points
        p2, p98 = np.percentile(enhanced, (2, 98))
        result = np.interp(enhanced, (p2, p98), (0, 255)).astype(np.uint8)
        
        return result

    def fix_rotation(self, image):
        """Rotation correction with proper padding"""
        self.logger.info("Correcting rotation")
        edges = cv2.Canny(image, 50, 150)
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
                rows, cols = image.shape
                center = (cols/2, rows/2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                
                # Calculate new dimensions
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((rows * sin) + (cols * cos))
                new_h = int((rows * cos) + (cols * sin))
                
                # Adjust transformation matrix
                M[0, 2] += (new_w/2) - center[0]
                M[1, 2] += (new_h/2) - center[1]
                
                # Apply rotation with border replication
                rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
                return rotated
        
        return image

    def fix_frequency_domain(self, image):
        """Enhanced frequency domain correction"""
        self.logger.info("Fixing frequency domain issues")
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        
        # Create high-pass filter
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        
        # Apply filter
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        result = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return result

    def remove_foreign_object(self, image):
        """Remove foreign objects using advanced vertical filtering"""
        self.logger.info("Removing foreign objects")
        
        # Initial bar width estimation
        edges = cv2.Canny(image, 50, 150)
        vertical_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
        
        if vertical_lines is not None:
            widths = []
            for line in vertical_lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10:  # Nearly vertical lines
                    widths.append(abs(x2 - x1))
            if widths:
                bar_width = max(3, int(np.median(widths)))
            else:
                bar_width = 5
        else:
            bar_width = 5
            
        self.logger.debug(f"Estimated bar width: {bar_width}")
        
        # Create multiple kernels of increasing size
        kernel1 = np.ones((image.shape[0], bar_width)) / image.shape[0]
        kernel2 = np.ones((image.shape[0], bar_width * 3)) / image.shape[0]
        kernel3 = np.ones((image.shape[0], bar_width * 5)) / image.shape[0]
        
        # Progressive filtering
        cleaned1 = cv2.filter2D(image, -1, kernel1)
        cleaned2 = cv2.filter2D(cleaned1, -1, kernel2)
        cleaned3 = cv2.filter2D(cleaned2, -1, kernel3)
        
        # Remove horizontal noise
        horizontal_kernel = np.ones((bar_width, 1)) / bar_width
        final_clean = cv2.filter2D(cleaned3, -1, horizontal_kernel)
        
        # Restore binary image
        _, result = cv2.threshold(final_clean, 127, 255, cv2.THRESH_BINARY)
        
        return result

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.logger.info("Starting preprocessing pipeline")
        self.steps = []
        self.add_step("Original Image", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Detect problems
        problems = []
        if self.detect_foreign_object(gray):
            problems.append('foreign_object')
        if self.detect_blur(gray):
            problems.append('blur')
        if self.detect_low_contrast(gray):
            problems.append('low_contrast')
        if self.detect_salt_pepper(gray):
            problems.append('salt_pepper')
        if self.detect_frequency_domain_issues(gray):
            problems.append('frequency_domain')
        if self.detect_rotation(gray):
            problems.append('rotation')
            
        self.logger.info(f"Detected problems: {problems}")
        
        processed = gray.copy()
        
        # Apply fixes in order
        if 'rotation' in problems:
            processed = self.fix_rotation(processed)
            self.add_step("Fixed Rotation", processed)
            
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed)
            self.add_step("Removed Foreign Object", processed)
            
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed)
            self.add_step("Fixed Frequency Domain", processed)
            
        if 'salt_pepper' in problems:
            processed = self.fix_salt_and_pepper(processed)
            self.add_step("Removed Salt & Pepper", processed)
            
        if 'blur' in problems:
            processed = self.fix_blur(processed)
            self.add_step("Fixed Blur", processed)
            
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed)
            self.add_step("Enhanced Contrast", processed)
            
        # Final cleanup
        _, binary = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)
        self.add_step("Final Result", binary)
        
        self.logger.info("Preprocessing completed")
        return binary

    def visualize_steps(self):
        """Visualize processing steps"""
        self.logger.info("Visualizing steps")
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