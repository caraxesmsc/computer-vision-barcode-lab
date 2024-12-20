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
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Create a unique log file for each run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/barcode_processor_{timestamp}.log'

        # Configure logging
        self.logger = logging.getLogger('BarcodeProcessor')
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))
        self.logger.info(f"Step added: {description}")
        self.logger.debug(f"Image shape: {image.shape}, Mean intensity: {np.mean(image):.2f}, Std: {np.std(image):.2f}")

    def estimate_bar_thickness(self, image):
        """Estimate barcode bar thickness using vertical runs of black pixels after binarization"""
        self.logger.info("Estimating bar thickness...")
        temp_binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 7
        )

        heights = []
        for col in range(temp_binary.shape[1]):
            column = temp_binary[:, col]
            run_length = 0
            black_runs = []
            for val in column:
                if val == 0:
                    run_length += 1
                else:
                    if run_length > 0:
                        black_runs.append(run_length)
                    run_length = 0
            if run_length > 0:
                black_runs.append(run_length)
            if black_runs:
                heights.extend(black_runs)

        if not heights:
            self.logger.warning("No bar heights detected, using default values")
            return 3, 9  # Default fallback

        min_width = max(1, int(np.percentile(heights, 10)))  # 10th percentile for min width
        pattern_width = max(3, int(np.median(heights)))  # Median for pattern width

        min_width = min(min_width, 10)
        pattern_width = min(pattern_width, 30)
        
        self.logger.info(f"Estimated min bar width: {min_width}, pattern width: {pattern_width}")
        return min_width, pattern_width

    def detect_compression_artifacts(self, gray):
        """Detect compression artifacts using block-based analysis"""
        self.logger.info("Checking for compression artifacts...")
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        block_size = 8

        if magnitude.shape[0] < block_size or magnitude.shape[1] < block_size:
            self.logger.warning("Image too small for compression artifact detection")
            return False

        blocks = cv2.boxFilter(magnitude, -1, (block_size, block_size))
        block_variation = np.std(blocks)
        
        has_artifacts = block_variation > 10
        self.logger.info(f"Compression artifacts detected: {has_artifacts} (variation: {block_variation:.2f})")
        return has_artifacts

    def detect_image_problems(self, image):
        """Detect image problems with detailed logging"""
        self.logger.info("Starting problem detection...")
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        if laplacian_var < 90 and edge_density < 0.15:
            problems.append('blur')
            self.logger.info(f"Blur detected (Laplacian var: {laplacian_var:.2f}, Edge density: {edge_density:.2f})")

        # Low contrast detection
        p5, p95 = np.percentile(gray, [5, 95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')
            self.logger.info(f"Low contrast detected (Range: {p95-p5:.2f})")

        # Salt and pepper detection
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        noise_level = np.percentile(noise_diff, 95)
        if noise_level > 40:
            problems.append('salt_pepper')
            self.logger.info(f"Salt & pepper noise detected (Level: {noise_level:.2f})")

        # Compression artifacts
        if self.detect_compression_artifacts(gray):
            problems.append('compression_artifacts')

        # Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        freq_std = np.std(magnitude_spectrum)
        if freq_std > 45:
            problems.append('frequency_domain')
            self.logger.info(f"Frequency domain issues detected (Std: {freq_std:.2f})")

        # Rotation detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(gray.shape) // 3)
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
                self.logger.info(f"Rotation detected (Median angle: {np.median(angles):.2f})")

        self.logger.info(f"Problems detected: {problems}")
        return problems

    def remove_foreign_object(self, image, min_bar_width):
        """Remove foreign objects using vertical kernel filtering"""
        self.logger.info(f"Removing foreign objects (min bar width: {min_bar_width})")

        # Create kernels
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_bar_width * 10))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1000))

        # Multi-stage filtering
        eroded = cv2.erode(image, kernel1, iterations=2)
        self.logger.debug("Applied initial erosion")

        dilated1 = cv2.dilate(eroded, kernel1, iterations=2)
        self.logger.debug("Applied first dilation")

        dilated2 = cv2.dilate(dilated1, kernel2, iterations=1)
        self.logger.debug("Applied second dilation")

        cleaned = cv2.dilate(dilated2, kernel3, iterations=1)
        self.logger.debug("Applied final dilation")

        return cleaned

    def fix_frequency_domain(self, image):
        """Fix frequency domain issues"""
        self.logger.info("Applying frequency domain fix")
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
        mask[mask_area] = 0
        
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        result = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.logger.debug("Frequency domain fix applied")
        return result

    def fix_blur(self, image):
        """Fix blur using two-stage approach"""
        self.logger.info("Fixing blur")
        # First stage
        denoised = cv2.medianBlur(image, 3)
        blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
        self.logger.debug("Applied initial denoising")
        
        # Second stage
        gaussian = cv2.GaussianBlur(blurred, (0, 0), 2)
        unsharp = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)
        self.logger.debug("Applied unsharp masking")
        
        return unsharp

    def fix_contrast(self, image):
        """Fix contrast using CLAHE and normalization"""
        self.logger.info("Enhancing contrast")
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        self.logger.debug(f"Normalized contrast range: {p2:.2f} - {p98:.2f}")
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        self.logger.debug("Applied CLAHE enhancement")
        
        return enhanced

    def final_cleanup(self, binary, min_bar_width):
        """Final cleanup using vertical pattern averaging"""
        self.logger.info("Performing final cleanup")
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]
        cleaned = np.zeros_like(binary)
        
        for i in range(0, binary.shape[1] - min_bar_width + 1):
            region = binary[:, i:i+min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i+min_bar_width] = (vertical_avg > 127).astype(np.uint8) * 255
        
        self.logger.debug("Applied vertical pattern averaging")
        return cleaned

    def preprocess_image(self, image):
        """Main preprocessing pipeline with logging"""
        self.logger.info("Starting image preprocessing")
        self.steps = []
        self.add_step("Original Image", image)

        # Initial grayscale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Problem detection
        problems = self.detect_image_problems(gray)
        min_bar_width, _ = self.estimate_bar_thickness(gray)
        
        processed = gray.copy()

        # Apply fixes based on detected problems
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed, min_bar_width)
            self.add_step("Removed Foreign Object", processed)

        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed)
            self.add_step("Fixed Frequency Domain", processed)

        if 'blur' in problems:
            processed = self.fix_blur(processed)
            self.add_step("Fixed Blur", processed)

        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed)
            self.add_step("Enhanced Contrast", processed)

        # Final cleanup
        cleaned = self.final_cleanup(processed, min_bar_width)
        self.add_step("Final Cleanup", cleaned)

        self.logger.info("Image preprocessing completed")
        return cleaned

    def visualize_steps(self):
        """Visualize processing steps"""
        self.logger.info("Visualizing processing steps")
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
        plt.savefig('steps.png')  # Save visualization
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