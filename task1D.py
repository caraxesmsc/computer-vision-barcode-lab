import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tkinter import filedialog

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def detect_image_problems(self, image):
        """Optimized problem detection"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Foreign object detection
        vertical_profile = np.mean(gray, axis=0)
        profile_std = np.std(vertical_profile)
        if np.any(np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * profile_std):
            problems.append('foreign_object')

        # Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:
            problems.append('blur')

        # Low contrast detection
        p5, p95 = np.percentile(gray, [5, 95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')

        # Salt and pepper detection
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.percentile(noise_diff, 95) > 30:
            problems.append('salt_pepper')

        # Frequency domain problem detection
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # Rotation detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
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
        """Remove foreign objects using vertical average filtering"""
        vertical_profile = np.mean(image, axis=0)
        smoothed_profile = np.convolve(vertical_profile, np.ones(5) / 5, mode='same')
        threshold = np.mean(smoothed_profile) + 2 * np.std(smoothed_profile)

        anomalous = vertical_profile > threshold
        cleaned = image.copy()

        for i in range(image.shape[1]):
            if anomalous[i]:
                left = max(0, i - 1)
                right = min(image.shape[1] - 1, i + 1)
                while left > 0 and anomalous[left]:
                    left -= 1
                while right < image.shape[1] - 1 and anomalous[right]:
                    right += 1
                if not anomalous[left]:
                    cleaned[:, i] = image[:, left]
                elif not anomalous[right]:
                    cleaned[:, i] = image[:, right]
        return cleaned

    def fix_blur(self, image):
        """Fix blur using unsharp masking"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return unsharp

    def fix_contrast(self, image):
        """Fix contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def fix_salt_pepper(self, image):
        """Fix salt and pepper noise using median filtering"""
        return cv2.medianBlur(image, 3)

    def fix_frequency_domain(self, image):
        """Fix frequency domain problems"""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2 <= r * r
        mask[mask_area] = 0
        f_shift *= mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def fix_rotation(self, image):
        """Fix rotation using HoughLines"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
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
                matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)
                return cv2.warpAffine(image, matrix, (cols, rows), flags=cv2.INTER_CUBIC)
        return image

    def binarize(self, image):
        """Binarize the image using adaptive thresholding"""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
        )

    def crop_to_barcode(self, binary):
        """Crop to the largest contour, assuming it is the barcode"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(binary.shape[1] - x, w + 2 * padding)
            h = min(binary.shape[0] - y, h + 2 * padding)
            return binary[y:y + h, x:x + w]
        return binary

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        # Apply fixes in order
        if 'foreign_object' in problems:
            gray = self.remove_foreign_object(gray)
            self.add_step("Removed Foreign Object", gray)

        if 'frequency_domain' in problems:
            gray = self.fix_frequency_domain(gray)
            self.add_step("Fixed Frequency Domain", gray)

        if 'salt_pepper' in problems:
            gray = self.fix_salt_pepper(gray)
            self.add_step("Removed Salt and Pepper", gray)

        if 'blur' in problems:
            gray = self.fix_blur(gray)
            self.add_step("Fixed Blur", gray)

        if 'low_contrast' in problems:
            gray = self.fix_contrast(gray)
            self.add_step("Enhanced Contrast", gray)

        binary = self.binarize(gray)
        self.add_step("Binarized", binary)

        if 'rotation' in problems:
            binary = self.fix_rotation(binary)
            self.add_step("Corrected Rotation", binary)

        cropped = self.crop_to_barcode(binary)
        self.add_step("Cropped", cropped)

        return cropped

    def visualize_steps(self):
        """Visualize all processing steps"""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (description, image) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.imshow(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
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
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()
