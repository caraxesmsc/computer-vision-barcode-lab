import cv2
import numpy as np
from scipy import ndimage, fftpack
import matplotlib.pyplot as plt
from tkinter import filedialog

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def detect_image_problems(self, image):
        """Improved analysis of image problems"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Improved blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 50  # Reduced from 100
        if laplacian_var < blur_threshold:
            problems.append('blur')

        # Enhanced contrast detection
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        cumsum = np.cumsum(hist_norm)
        idx_5 = np.where(cumsum >= 0.05)[0][0]
        idx_95 = np.where(cumsum >= 0.95)[0][0]
        if (idx_95 - idx_5) < 100 or np.mean(gray) < 50 or np.mean(gray) > 200:
            problems.append('low_contrast')

        # Improved salt and pepper detection
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.percentile(noise_diff, 95) > 30:  # Changed from mean to 95th percentile
            problems.append('salt_pepper')

        # Enhanced compression artifact detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        if 0.1 < edge_density < 0.3:  # Typical range for compression artifacts
            problems.append('compression_artifacts')

        # Improved rotation detection
        # Only detect rotation if we have clear edges
        if edge_density > 0.05:  # Ensure we have enough edges
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(gray.shape) * 0.3))
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta)
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                if angles and abs(np.median(angles)) > 3:  # Reduced from 5
                    problems.append('rotation')

        # Frequency domain problem detection
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 50:
            problems.append('frequency_domain')

        return problems

    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        # Detect problems
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Handle frequency domain problems first
        if 'frequency_domain' in problems:
            gray = self.fix_frequency_domain(gray)
            self.add_step("Fixed Frequency Domain Issues", gray)

        # Enhanced preprocessing sequence
        if 'blur' in problems:
            gray = self.fix_blur(gray)
            self.add_step("Fixed Blur", gray)

        if 'salt_pepper' in problems:
            gray = self.fix_salt_pepper(gray)
            self.add_step("Removed Salt and Pepper Noise", gray)

        if 'compression_artifacts' in problems:
            gray = self.fix_compression(gray)
            self.add_step("Reduced Compression Artifacts", gray)

        if 'low_contrast' in problems:
            gray = self.fix_contrast(gray)
            self.add_step("Enhanced Contrast", gray)

        # Improved binarization
        binary = self.binarize(gray)
        self.add_step("Binarized Image", binary)

        # Post-binarization cleanup
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.add_step("Cleaned Binary Image", binary)

        if 'rotation' in problems:
            binary = self.fix_rotation(binary)
            self.add_step("Corrected Rotation", binary)

        # Improved cropping
        cropped = self.crop_to_barcode(binary)
        self.add_step("Cropped Barcode", cropped)

        return cropped

    @staticmethod
    def fix_blur(image):
        # Enhanced unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2)
        unsharp = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        return cv2.normalize(unsharp, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def fix_salt_pepper(image):
        # Progressive median filtering
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 3)
        return result

    @staticmethod
    def fix_contrast(image):
        # Enhanced contrast correction
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        # Additional normalization
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced

    @staticmethod
    def fix_compression(image):
        # Enhanced compression artifact removal
        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return cv2.GaussianBlur(denoised, (3, 3), 0)

    @staticmethod
    def fix_frequency_domain(image):
        # Frequency domain filtering
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
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
        
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def binarize(image):
        # Enhanced adaptive thresholding
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 85, 15
        )
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    @staticmethod
    def fix_rotation(image):
        # Enhanced rotation correction
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(image.shape) * 0.3))
        
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
                return cv2.warpAffine(image, matrix, (cols, rows), 
                                    flags=cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
        
        return image

    @staticmethod
    def crop_to_barcode(binary):
        # Enhanced cropping with aspect ratio check
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return binary
            
        # Filter contours by aspect ratio and area
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = w * h
            if 1.5 < aspect_ratio < 5 and area > (binary.shape[0] * binary.shape[1] * 0.05):
                valid_contours.append((contour, area))
                
        if not valid_contours:
            return binary
            
        # Use the largest valid contour
        largest_contour = max(valid_contours, key=lambda x: x[1])[0]
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add proportional padding
        padding_x = int(0.1 * w)
        padding_y = int(0.1 * h)
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(binary.shape[1] - x, w + 2 * padding_x)
        h = min(binary.shape[0] - y, h + 2 * padding_y)
        
        return binary[y:y+h, x:x+w]

    def visualize_steps(self):
        """Visualize all processing steps"""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (description, image) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.imshow(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                     cmap='gray')
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