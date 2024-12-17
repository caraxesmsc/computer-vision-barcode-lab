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
        """Detect problems combining strategies from codebase 1 and codebase 2."""
        problems = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

        # Foreign object detection (from codebase 2)
        vertical_profile = np.mean(gray, axis=0)
        profile_std = np.std(vertical_profile)
        if np.any(np.abs(vertical_profile - np.mean(vertical_profile)) > 2.5 * profile_std):
            problems.append('foreign_object')

        # Blur detection (from codebase 1)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:  # Adjusted threshold from codebase 1 was 100, codebase 2 was 30
            problems.append('blur')

        # Low contrast detection (from codebase 1)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()
        cumsum = np.cumsum(hist_norm)
        idx_5 = np.where(cumsum>=0.05)[0][0]
        idx_95 = np.where(cumsum>=0.95)[0][0]
        if (idx_95 - idx_5) < 50:
            problems.append('low_contrast')

        # Salt and pepper detection (from codebase 1)
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.mean(noise_diff) > 10:
            problems.append('salt_pepper')

        # Frequency domain problem detection (from codebase 2)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        # If std too low or too high can indicate issues.
        # We'll say if std < some threshold indicates uniform freq? Codebase2 used std >45 for FD problem:
        # Let's choose a heuristic: if std of magnitude < 10 or > 150 might be suspicious. We'll use codebase 2 logic:
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # Rotation detection (from codebase 1)
        edges = cv2.Canny(gray, 100, 200)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        return problems

    def remove_foreign_object(self, image, window_size=5):
        """Vertical average filter approach from codebase 2 to remove foreign objects."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        vertical_profile = np.mean(gray, axis=0)
        smoothed_profile = np.convolve(vertical_profile, np.ones(window_size)/window_size, mode='same')
        threshold = np.mean(smoothed_profile) + 1.5 * np.std(smoothed_profile)
        
        anomalous = vertical_profile > threshold
        cleaned = gray.copy()
        for i in range(gray.shape[1]):
            if anomalous[i]:
                left_idx = i-1
                right_idx = i+1
                # Move left_idx left until non-anomalous or start
                while left_idx >= 0 and anomalous[left_idx]:
                    left_idx -= 1
                # Move right_idx right until non-anomalous or end
                while right_idx < gray.shape[1] and anomalous[right_idx]:
                    right_idx += 1
                # Choose nearest good column
                if left_idx >= 0 and not anomalous[left_idx]:
                    cleaned[:, i] = gray[:, left_idx]
                elif right_idx < gray.shape[1] and not anomalous[right_idx]:
                    cleaned[:, i] = gray[:, right_idx]
                # If both sides anomalous, do nothing special (rare case)
        
        return cleaned

    def fix_frequency_domain(self, image):
        """Fix frequency domain issues from codebase 2."""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # radius for low-frequency suppression
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
        mask[mask_area] = 0
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def remove_salt_and_pepper(self, image):
        """Enhanced salt and pepper removal (iterative median from codebase 2 idea)."""
        # We'll do multiple passes of median filter
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        # If still noisy, one more pass
        noise_level = np.std(result - image)
        if noise_level > 30:
            result = cv2.medianBlur(result, 3)
        return result

    @staticmethod
    def fix_blur(image):
        """From codebase 1: unsharp masking to fix blur."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return unsharp

    @staticmethod
    def fix_contrast(image):
        """From codebase 1: CLAHE to improve contrast."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def binarize(image):
        """From codebase 1: Adaptive thresholding."""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 10
        )

    @staticmethod
    def fix_rotation(image):
        """From codebase 1: Rotate based on HoughLines analysis."""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = []
            for rho, theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle-180)
            if angles:
                median_angle = np.median(angles)
                rows, cols = image.shape
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                return cv2.warpAffine(image, matrix, (cols, rows), flags=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def crop_to_barcode(binary):
        """From codebase 1: Crop to largest contour (barcode region)."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary.shape[1] - x, w + 2 * padding)
        h = min(binary.shape[0] - y, h + 2 * padding)
        return binary[y:y + h, x:x + w]

    def preprocess_image(self, image):
        """Unified preprocessing pipeline."""
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        # Apply fixes in order:

        # Foreign object first
        if 'foreign_object' in problems:
            gray = self.remove_foreign_object(gray)
            self.add_step("Removed Foreign Object", gray)

        # Frequency domain second
        if 'frequency_domain' in problems:
            gray = self.fix_frequency_domain(gray)
            self.add_step("Fixed Frequency Domain", gray)

        # Salt and pepper third
        if 'salt_pepper' in problems:
            gray = self.remove_salt_and_pepper(gray)
            self.add_step("Removed Salt & Pepper", gray)

        # Blur fourth
        if 'blur' in problems:
            gray = self.fix_blur(gray)
            self.add_step("Fixed Blur", gray)

        # Contrast fifth
        if 'low_contrast' in problems:
            gray = self.fix_contrast(gray)
            self.add_step("Enhanced Contrast", gray)

        # Binarize now
        binary = self.binarize(gray)
        self.add_step("Binarized Image", binary)

        # Fix rotation
        if 'rotation' in problems:
            binary = self.fix_rotation(binary)
            self.add_step("Corrected Rotation", binary)

        # Crop
        cropped = self.crop_to_barcode(binary)
        self.add_step("Cropped Barcode", cropped)

        return cropped

    def visualize_steps(self):
        """Visualize all processing steps."""
        fig, axes = plt.subplots(1, len(self.steps), figsize=(5*len(self.steps), 5))
        if len(self.steps) == 1:
            axes = [axes]
        for i, (description, image) in enumerate(self.steps):
            ax = axes[i]
            ax.imshow(image if len(image.shape)==2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
            ax.set_title(description, fontsize=8)
            ax.axis('off')
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
