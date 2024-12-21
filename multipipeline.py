import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodeImageQuality:
    """Class to evaluate barcode image quality"""
    @staticmethod
    def calculate_edge_quality(image: np.ndarray) -> float:
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.mean(edges > 0)
        edge_coherence = cv2.Laplacian(image, cv2.CV_64F).var()
        return edge_density * 100 + edge_coherence * 0.1

    @staticmethod
    def calculate_contrast_quality(image: np.ndarray) -> float:
        p5, p95 = np.percentile(image, [5, 95])
        contrast_range = p95 - p5
        return contrast_range

    @staticmethod
    def calculate_pattern_quality(image: np.ndarray) -> float:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        pattern_strength = np.mean(np.abs(sobelx))
        return pattern_strength

    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        noise = np.abs(image.astype(float) - blurred.astype(float))
        return 255 - np.mean(noise)

    @classmethod
    def evaluate_quality(cls, image: np.ndarray) -> float:
        if image is None:
            return 0.0
        
        edge_score = cls.calculate_edge_quality(image)
        contrast_score = cls.calculate_contrast_quality(image)
        pattern_score = cls.calculate_pattern_quality(image)
        noise_score = cls.calculate_noise_quality(image)
        
        # Weighted combination
        return (edge_score * 0.4 + 
                contrast_score * 0.3 + 
                pattern_score * 0.2 + 
                noise_score * 0.1)




class GeneralPipeline1:
    """Pipeline 1: Based on codebase 3's approach"""
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def detect_barcode_region(self, image):
        """Enhanced barcode region detection using Sobel and adaptive thresholding"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        # Adaptive thresholding to find regions with high concentration of vertical lines
        thresh = cv2.adaptiveThreshold(gradx, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

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

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Apply initial denoising
        denoised = cv2.medianBlur(gray, 3)
        self.add_step("Denoised", denoised)

        # Detect the barcode region
        region = self.detect_barcode_region(denoised)
        if region is None:
            print("Barcode region not detected in Pipeline 1.")
            return denoised
        x, y, w, h = region

        # Crop the region of interest
        roi = denoised[y:y+h, x:x+w]
        self.add_step("ROI", roi)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)
        self.add_step("Enhanced Contrast", enhanced)

        # Binarize the enhanced region
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.add_step("Binarized", binary)

        # Perform morphological operations to clean up the barcode
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.add_step("Cleaned", cleaned)

        return cleaned

class GeneralPipeline2:
    """Pipeline 2: Based on codebase 6's morphological approach"""
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))
        
    def detect_barcode_region(self, image):
        """Enhanced barcode region detection"""
        # Use Sobel edge detection to enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        # Apply Gaussian blur and Otsu's thresholding
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by size, aspect ratio, and density of vertical lines
        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
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

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Detect the barcode region
        region = self.detect_barcode_region(gray)
        if region is None:
            print("Barcode region not detected in Pipeline 2.")
            return gray
            
        x, y, w, h = region
        roi = gray[y:y+h, x:x+w]

        # Apply morphological operations for noise reduction
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        self.add_step("Morphological Opening", opening)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(opening)
        self.add_step("CLAHE Enhanced", enhanced)

        # Binarize the image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.add_step("Binarized", binary)

        return binary

class GeneralPipeline3:
    """Pipeline 3: Based on codebase 21's quality-based approach"""
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))
        
    def estimate_bar_thickness(self, image):
        """Estimate barcode bar thickness using vertical runs of black pixels after binarization."""
        temp_binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 10
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
            return 5  # Default fallback if no bars are detected
        return int(np.median(heights))

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Detect problems and barcode region
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if barcode_region is None:
            print("Barcode region not detected for any processing.")
            return gray

        processed = gray.copy()
        x, y, w, h = barcode_region

        # Apply fixes in optimal order
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
            processed = self.remove_salt_and_pepper(processed, barcode_region)
            self.add_step("Removed Salt & Pepper", processed)

        if 'blur' in problems:
            processed = self.fix_blur(processed, barcode_region)
            self.add_step("Fixed Blur", processed)

        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed, barcode_region)
            self.add_step("Enhanced Contrast", processed)

        # Extract ROI for final processing
        x, y, w, h = barcode_region
        roi = processed[y:y + h, x:x + w]

        # Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        # Final cleanup using vertical averaging
        cleaned = self.final_barcode_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        return cleaned

    def final_barcode_cleanup(self, binary):
        """Clean up barcode after all processing using vertical averaging based on estimated bar thickness."""
        min_bar_width = self.estimate_bar_thickness(binary)
        print(f"Estimated bar thickness in final_barcode_cleanup: {min_bar_width}")

        # Ensure that min_bar_width is a positive integer
        min_bar_width = max(1, int(min_bar_width))

        # Create vertical averaging kernel
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]

        # Apply kernel along horizontal axis
        cleaned = np.zeros_like(binary)
        for i in range(0, binary.shape[1] - min_bar_width + 1):
            region = binary[:, i:i + min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i + min_bar_width] = (vertical_avg > 127).astype(np.uint8) * 255

        return cleaned

    def fix_contrast(self, image, barcode_region, contrast_type=None):
        """Fix contrast using CLAHE and gamma correction"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)

        # Apply gamma correction
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result
    
    def fix_blur(self, image, barcode_region):
        """Fix blur using a two-stage approach: unsharp masking and sharpening kernel"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.5, gaussian, -0.5, 0)

        # Second stage: Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        # Check if further sharpening is needed
        laplacian_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if laplacian_var < 150:  # Only sharpen more if still somewhat blurry
            sharpened = cv2.filter2D(unsharp, -1, kernel)
        else:
            sharpened = unsharp

        result = image.copy()
        result[y:y+h, x:x+w] = sharpened
        return result

    def remove_salt_and_pepper(self, image, barcode_region):
        """Remove salt and pepper noise using median filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Apply median filtering
        filtered = cv2.medianBlur(roi, 3)

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
            h=5,  # Filter strength - lower than usual to preserve barcode details
            templateWindowSize=7,
            searchWindowSize=21
        )

        result = image.copy()
        result[y:y+h, x:x+w] = denoised
        return result
    
    def fix_frequency_domain(self, image, barcode_region):
        """Fix frequency domain issues using a Gaussian high-pass filter"""
        x, y, w, h = barcode_region
        roi = image[y:y + h, x:x + w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)

        # Create a Gaussian high-pass filter
        rows, cols = roi.shape
        crow, ccol = rows // 2, cols // 2
        x = np.linspace(-ccol, ccol - 1, cols)
        y = np.linspace(-crow, crow - 1, rows)
        xv, yv = np.meshgrid(x, y)
        sigma = 30  # Adjust sigma for filter strength
        gaussian_filter = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        mask = 1 - gaussian_filter  # Invert to make it high-pass

        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y:y + h, x:x + w] = cleaned
        return result

    def detect_salt_pepper(self, roi):
        """Detect salt and pepper noise using median filtering and thresholding."""
        median = cv2.medianBlur(roi, 3)
        diff = np.abs(roi - median)
        return np.percentile(diff, 95) > 40

    def detect_compression_artifacts(self, gray):
        """Heuristic detection of compression artifacts based on blocky high-frequency patterns."""
        # Use Sobel filters to check blocky patterns
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Check variation in block-averaged magnitude
        block_size = 8
        blurred = cv2.boxFilter(magnitude.astype(np.float32), -1, (block_size,block_size))
        block_variation = np.std(blurred)
        # Adjust threshold as needed
        return block_variation > 10

    def detect_image_problems(self, image):
        """Detect image problems, with barcode region detection and pattern analysis."""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Find barcode region for focused analysis
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]

        # Estimate bar width and pattern width using the ROI
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

        # Low contrast detection
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')

        # Salt and pepper detection
        if self.detect_salt_pepper(roi):
            problems.append('salt_pepper')

        # Compression artifacts detection
        if self.detect_compression_artifacts(roi):
            problems.append('compression_artifacts')

        # Frequency domain issues
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # Rotation detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(w, h) // 3)
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

        self.problems = problems
        return problems, barcode_region

    def binarize(self, image):
        """Binarize the image using adaptive thresholding with pre-normalization"""
        # Normalize contrast
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)

        # Apply adaptive thresholding with carefully chosen parameters
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Slightly smaller window to prevent holes
            5    # Adjusted offset
        )

        # Light cleanup without creating holes
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def final_barcode_cleanup(self, binary):
        """Clean up barcode after all processing using vertical averaging based on estimated bar thickness."""
        min_bar_width, _ = self.estimate_bar_thickness(binary)
        
        # Ensure min_bar_width is at least 1
        min_bar_width = max(1, min_bar_width)

        # Create vertical averaging kernel
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]

        # Apply kernel along horizontal axis
        cleaned = np.zeros_like(binary)
        for i in range(0, binary.shape[1] - min_bar_width + 1):
            region = binary[:, i:i + min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i + min_bar_width] = (vertical_avg > 127).astype(np.uint8) * 255

        return cleaned

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
            print("Barcode region not found.")
            return gray

        processed = gray.copy()
        x, y, w, h = barcode_region

        # Apply fixes in optimal order

        if 'rotation' in problems:
            processed = self.fix_rotation(processed, barcode_region)
            self.add_step("Fixed Rotation", processed)
            # Update barcode region after rotation
            barcode_region = self.detect_barcode_region(processed)
            if barcode_region is None:
                print("Barcode region lost after rotation.")
                return processed  # Return if barcode region is lost after rotation
            x, y, w, h = barcode_region

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
            processed = self.remove_salt_and_pepper(processed, barcode_region)
            self.add_step("Removed Salt & Pepper", processed)

        if 'blur' in problems:
            processed = self.fix_blur(processed, barcode_region)
            self.add_step("Fixed Blur", processed)

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
        cleaned = self.final_barcode_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        return cleaned

    def visualize_steps(self):
        """Enhanced visualization with proper subplot handling"""
        n_steps = len(self.steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
        # Flatten axes array for easier indexing
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

class BarcodeProcessor:
    def visualize_steps(self):
        """Enhanced visualization with proper subplot handling"""
        n_steps = len(self.steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
        # Flatten axes array for easier indexing
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


if __name__ == "__main__":
    preprocessor = GeneralPipeline1()
    preprocessor = BarcodeProcessor()
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()
