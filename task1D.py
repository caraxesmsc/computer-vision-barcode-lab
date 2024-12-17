import cv2
import numpy as np
from scipy import ndimage, fftpack
import matplotlib.pyplot as plt
from tkinter import filedialog

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        self.steps.append((description, image.copy()))

    def detect_image_problems(self, image):
        """Optimized problem detection combining best elements"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Foreign object detection (improved)
        vertical_profile = np.mean(gray, axis=0)
        horizontal_profile = np.mean(gray, axis=1)
        v_thresh = np.mean(vertical_profile) + 2.5 * np.std(vertical_profile)
        h_thresh = np.mean(horizontal_profile) + 2.5 * np.std(horizontal_profile)
        
        v_anomaly = np.any(vertical_profile > v_thresh)
        h_anomaly = np.any(horizontal_profile > h_thresh)
        if v_anomaly or h_anomaly:
            # Additional check: verify if the anomaly is localized
            if np.sum((vertical_profile > v_thresh).astype(int)) < gray.shape[1] * 0.4:
                problems.append('foreign_object')

        # Blur detection (optimized)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Higher threshold from codebase 1
            problems.append('blur')

        # Enhanced contrast detection
        p5, p95 = np.percentile(gray, [5, 95])
        mean_intensity = np.mean(gray)
        if (p95 - p5) < 80 or mean_intensity < 50 or mean_intensity > 200:
            problems.append('low_contrast')

        # Salt and pepper detection (improved)
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.percentile(noise_diff, 95) > 30:
            problems.append('salt_pepper')

        # Frequency domain analysis (refined)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # Rotation detection (from codebase 1 - most reliable)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
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
                problems.append('rotation')
        
        return problems

    def remove_foreign_object(self, image):
        """Enhanced foreign object removal"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Identify anomalous regions
        vertical_profile = np.mean(gray, axis=0)
        horizontal_profile = np.mean(gray, axis=1)
        
        # Calculate dynamic thresholds
        v_thresh = np.mean(vertical_profile) + 2 * np.std(vertical_profile)
        h_thresh = np.mean(horizontal_profile) + 2 * np.std(horizontal_profile)
        
        # Create masks
        v_mask = vertical_profile > v_thresh
        h_mask = horizontal_profile > h_thresh
        
        # Apply corrections
        cleaned = gray.copy()
        
        # Fix vertical anomalies
        for i in range(gray.shape[1]):
            if v_mask[i]:
                left = max(0, i-1)
                right = min(gray.shape[1]-1, i+1)
                while left > 0 and v_mask[left]:
                    left -= 1
                while right < gray.shape[1]-1 and v_mask[right]:
                    right += 1
                if not v_mask[left]:
                    cleaned[:, i] = gray[:, left]
                elif not v_mask[right]:
                    cleaned[:, i] = gray[:, right]
        
        # Fix horizontal anomalies
        for i in range(gray.shape[0]):
            if h_mask[i]:
                top = max(0, i-1)
                bottom = min(gray.shape[0]-1, i+1)
                while top > 0 and h_mask[top]:
                    top -= 1
                while bottom < gray.shape[0]-1 and h_mask[bottom]:
                    bottom += 1
                if not h_mask[top]:
                    cleaned[i, :] = gray[top, :]
                elif not h_mask[bottom]:
                    cleaned[i, :] = gray[bottom, :]
                    
        return cleaned

    def fix_blur(self, image):
        """Two-stage deblurring"""
        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        # Second stage: Sharpening kernel
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        return cv2.filter2D(unsharp, -1, kernel)

    def fix_contrast(self, image):
        """Multi-stage contrast enhancement"""
        # First stage: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Second stage: Contrast stretching
        p2, p98 = np.percentile(enhanced, (2, 98))
        return np.interp(enhanced, (p2, p98), (0, 255)).astype(np.uint8)

    def fix_salt_pepper(self, image):
        """Progressive median filtering"""
        # First pass with small kernel
        result = cv2.medianBlur(image, 3)
        
        # Second pass with larger kernel
        result = cv2.medianBlur(result, 5)
        
        # Check noise level
        if np.std(result - image) > 30:
            result = cv2.medianBlur(result, 3)
            
        return result

    def fix_frequency_domain(self, image):
        """Optimized frequency domain correction"""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        
        # Create frequency mask
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # Radius for frequency cutoff
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        
        # Apply mask and reconstruct
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def fix_rotation(self, image):
        """Rotation correction from codebase 1"""
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
                return cv2.warpAffine(image, matrix, (cols, rows), flags=cv2.INTER_CUBIC)
        return image

    def binarize(self, image):
        """Enhanced binarization to prevent holes"""
        # Pre-normalize
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        # Adaptive threshold with larger window
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            81,  # Increased window size
            10
        )
        
        # Morphological operations to close small holes
        kernel = np.ones((2,2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def crop_to_barcode(self, binary):
        """Enhanced cropping with aspect ratio validation"""
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
        
        if valid_contours:
            largest_contour = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding_x = int(0.1 * w)
            padding_y = int(0.1 * h)
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(binary.shape[1] - x, w + 2 * padding_x)
            h = min(binary.shape[0] - y, h + 2 * padding_y)
            return binary[y:y+h, x:x+w]
        
        return binary

    def preprocess_image(self, image):
        """Optimized preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Detect problems
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)
        
        # Process image
        processed = gray.copy()
        
        # Apply fixes in optimal order
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed)
            self.add_step("Removed Foreign Object", processed)
        
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed)
            self.add_step("Fixed Frequency Domain", processed)
        
        if 'salt_pepper' in problems:
            processed = self.fix_salt_pepper(processed)
            self.add_step("Removed Salt and Pepper", processed)
        
        if 'blur' in problems:
            processed = self.fix_blur(processed)
            self.add_step("Fixed Blur", processed)
        
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed)
            self.add_step("Enhanced Contrast", processed)
        
        # Binarize after all intensity corrections
        binary = self.binarize(processed)
        self.add_step("Binarized", binary)
        
        # Fix rotation
        if 'rotation' in problems:
            binary = self.fix_rotation(binary)
            self.add_step("Fixed Rotation", binary)
        
        # Crop last
        cropped = self.crop_to_barcode(binary)
        self.add_step("Cropped", cropped)
        
        return cropped

    def visualize_steps(self):
        """Enhanced visualization"""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
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