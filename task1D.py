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
        # Slightly increase threshold to reduce false positives
        if np.any(np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * profile_std):
            problems.append('foreign_object')

        # Blur detection (from codebase 1)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:  # slightly relaxed threshold from codebase 1 (100 to 80)
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
        """Vertical average filter approach (from codebase 2) to remove foreign objects."""
        gray = image if len(image.shape)==2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vertical_profile = np.mean(gray, axis=0)
        smoothed_profile = np.convolve(vertical_profile, np.ones(window_size)/window_size, mode='same')
        threshold = np.mean(smoothed_profile) + 1.5 * np.std(smoothed_profile)

        anomalous = vertical_profile > threshold
        cleaned = gray.copy()
        for i in range(gray.shape[1]):
            if anomalous[i]:
                left_idx = i-1
                right_idx = i+1
                while left_idx > 0 and anomalous[left_idx]:
                    left_idx -= 1
                while right_idx < gray.shape[1]-1 and anomalous[right_idx]:
                    right_idx += 1
                if left_idx >= 0 and not anomalous[left_idx]:
                    cleaned[:, i] = gray[:, left_idx]
                elif right_idx < gray.shape[1] and not anomalous[right_idx]:
                    cleaned[:, i] = gray[:, right_idx]
                # If both sides are anomalous, leave as is.
        return cleaned

    def fix_frequency_domain(self, image):
        """Frequency domain fix from codebase 2."""
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

        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def remove_salt_and_pepper(self, image):
        """Enhanced salt and pepper removal (from codebase 2)."""
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        # Check residual noise
        noise_level = np.std(result - image)
        if noise_level > 30:
            result = cv2.medianBlur(result, 3)
        return result

    @staticmethod
    def fix_blur(image):
        """From codebase 1."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return unsharp

    @staticmethod
    def fix_contrast(image):
        """From codebase 1."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def binarize(image):
        """From codebase 1."""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 10
        )

    @staticmethod
    def fix_rotation(image):
        """From codebase 1."""
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
        """From codebase 1."""
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
        """Unified preprocessing pipeline (based on codebase 3's approach)."""
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        processed = gray.copy()

        # Order of fixes as in codebase 3:
        # 1. Foreign object
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed)
            self.add_step("Removed Foreign Object", processed)

        # 2. Frequency domain
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed)
            self.add_step("Fixed Frequency Domain", processed)

        # 3. Salt and pepper
        if 'salt_pepper' in problems:
            processed = self.remove_salt_and_pepper(processed)
            self.add_step("Removed Salt & Pepper", processed)

        # 4. Blur
        if 'blur' in problems:
            processed = self.fix_blur(processed)
            self.add_step("Fixed Blur", processed)

        # 5. Compression artifacts
        if 'compression_artifacts' in problems:
            processed = self.fix_compression(processed)
            self.add_step("Fixed Compression Artifacts", processed)

        # 6. Low contrast
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed)
            self.add_step("Enhanced Contrast", processed)

        # Binarize
        binary = self.binarize(processed)
        self.add_step("Binarized Image", binary)

        # Rotation
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
        for i, (description, img) in enumerate(self.steps):
            ax = axes[i]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
