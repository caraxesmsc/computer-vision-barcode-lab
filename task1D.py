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
        """Enhanced problem detection with corrected thresholds"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

        # Foreign object detection using vertical and horizontal profiles
        vertical_profile = np.mean(gray, axis=0)
        horizontal_profile = np.mean(gray, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        
        if (np.any(np.abs(vertical_profile - np.mean(vertical_profile)) > 3 * v_std) or
            np.any(np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3 * h_std)):
            problems.append('foreign_object')

        # Improved blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # Adjusted threshold
            problems.append('blur')

        # Enhanced contrast detection
        p5, p95 = np.percentile(gray, [5, 95])
        if (p95 - p5) < 100:  # Adjusted threshold
            problems.append('low_contrast')

        # Improved salt and pepper detection
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.percentile(noise_diff, 95) > 30:
            problems.append('salt_pepper')

        # Compression artifacts detection
        edges = cv2.Canny(gray, 100, 200)
        if np.mean(edges) > 20:
            problems.append('compression_artifacts')

        # Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 40:
            problems.append('frequency_domain')

        # Rotation detection (from codebase 1 - more reliable)
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

    def remove_foreign_object(self, image, window_size=5):
        """Enhanced foreign object removal"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Combine vertical and horizontal filtering
        def filter_direction(img, axis):
            profile = np.mean(img, axis=axis)
            smoothed = np.convolve(profile, np.ones(window_size)/window_size, mode='same')
            threshold = np.mean(smoothed) + 2 * np.std(smoothed)
            anomalous = profile > threshold
            
            cleaned = img.copy()
            if axis == 0:  # Vertical
                for i in range(img.shape[1]):
                    if anomalous[i]:
                        left = max(0, i-1)
                        right = min(img.shape[1]-1, i+1)
                        while anomalous[left] and left > 0:
                            left -= 1
                        while anomalous[right] and right < img.shape[1]-1:
                            right += 1
                        if not anomalous[left]:
                            cleaned[:, i] = img[:, left]
                        elif not anomalous[right]:
                            cleaned[:, i] = img[:, right]
            else:  # Horizontal
                for i in range(img.shape[0]):
                    if anomalous[i]:
                        top = max(0, i-1)
                        bottom = min(img.shape[0]-1, i+1)
                        while anomalous[top] and top > 0:
                            top -= 1
                        while anomalous[bottom] and bottom < img.shape[0]-1:
                            bottom += 1
                        if not anomalous[top]:
                            cleaned[i, :] = img[top, :]
                        elif not anomalous[bottom]:
                            cleaned[i, :] = img[bottom, :]
            
            return cleaned

        # Apply both vertical and horizontal filtering
        cleaned_v = filter_direction(gray, axis=0)
        cleaned_h = filter_direction(gray, axis=1)
        
        # Combine results
        cleaned = np.minimum(cleaned_v, cleaned_h)
        
        return cleaned

    def fix_blur(self, image):
        """Enhanced deblurring"""
        # Unsharp masking with adjusted parameters
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        # Additional sharpening kernel
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(unsharp, -1, kernel)
        
        return sharpened

    def fix_contrast(self, image):
        """Enhanced contrast correction"""
        # CLAHE with optimal parameters
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Additional contrast stretching
        p2, p98 = np.percentile(enhanced, (2, 98))
        enhanced = np.interp(enhanced, (p2, p98), (0, 255)).astype(np.uint8)
        
        return enhanced

    def fix_salt_pepper(self, image):
        """Improved salt and pepper noise removal"""
        # Progressive median filtering
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        
        # Check residual noise
        noise_level = np.std(result - image)
        if noise_level > 30:
            result = cv2.medianBlur(result, 3)
        
        return result

    def fix_compression(self, image):
        """Enhanced compression artifact removal"""
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def fix_frequency_domain(self, image):
        """Frequency domain correction"""
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

    def fix_rotation(self, image):
        """Rotation correction from codebase 1 (more reliable)"""
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
        """Improved binarization without holes"""
        # Normalize contrast first
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        # Multi-level thresholding
        binary = cv2.adaptiveThreshold(
            normalized, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            101,  # Larger window
            10
        )
        
        # Clean up without creating holes
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def crop_to_barcode(self, binary):
        """Enhanced barcode region detection and cropping"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary
            
        # Filter valid contours based on aspect ratio and area
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
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Detect problems
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)
        
        # Process the image based on detected problems
        processed = gray.copy()
        
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
        
        if 'compression_artifacts' in problems:
            processed = self.fix_compression(processed)
            self.add_step("Fixed Compression Artifacts", processed)
        
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed)
            self.add_step("Enhanced Contrast", processed)
        
        # Binarization
        binary = self.binarize(processed)
        self.add_step("Binarized", binary)
        
        # Fix rotation if needed
        if 'rotation' in problems:
            binary = self.fix_rotation(binary)
            self.add_step("Fixed Rotation", binary)
        
        # Crop to barcode region
        cropped = self.crop_to_barcode(binary)
        self.add_step("Cropped", cropped)
        
        return cropped

    def visualize_steps(self):
        """Visualize all processing steps"""
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