import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    # -------------------------
    # Problem Detection
    # -------------------------
    def detect_image_problems(self, image):
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

        # Foreign object detection (vertical anomaly)
        vertical_profile = np.mean(gray, axis=0)
        v_std = np.std(vertical_profile)
        if np.any(np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * v_std):
            problems.append('foreign_object')

        # Blur detection (from codebase 1)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:
            problems.append('blur')

        # Low contrast detection (from codebase 1)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()
        cumsum = np.cumsum(hist_norm)
        idx_5 = np.where(cumsum>=0.05)[0][0]
        idx_95 = np.where(cumsum>=0.95)[0][0]
        if (idx_95 - idx_5) < 50:
            problems.append('low_contrast')

        # Salt and pepper detection
        median = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray - median)
        if np.mean(noise_diff) > 10:
            problems.append('salt_pepper')

        # Frequency domain detection
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # Rotation detection
        edges = cv2.Canny(gray, 100, 200)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        return problems

    # -------------------------
    # Utility Functions
    # -------------------------
    def fix_frequency_domain(self, image):
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
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        noise_level = np.std(result - image)
        if noise_level > 30:
            result = cv2.medianBlur(result, 3)
        return result

    def fix_blur(self, image):
        gaussian = cv2.GaussianBlur(image, (0,0), 3)
        unsharp = cv2.addWeighted(image, 1.3, gaussian, -0.3, 0)  # less aggressive
        # If still not sharp enough, second pass:
        lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if lap_var < 80:
            # another mild sharpening
            gaussian2 = cv2.GaussianBlur(unsharp, (0,0), 1)
            unsharp = cv2.addWeighted(unsharp, 1.2, gaussian2, -0.2, 0)
        return unsharp

    def fix_contrast(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        return enhanced

    def fix_rotation(self, image):
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

    def binarize(self, image):
        # Slightly smaller window or different offset to reduce holes
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 7
        )
        # Morph close to reduce holes
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def crop_to_barcode(self, binary):
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

    def estimate_bar_thickness(self, image):
        # We assume image is fairly preprocessed (contrast/frequency fixed)
        # Temporarily binarize image
        temp_binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 61, 10
        )
        # Find vertical black runs
        # For each column, find runs of 0 pixels
        heights = []
        for col in range(temp_binary.shape[1]):
            column = temp_binary[:, col]
            black_runs = []
            run_length = 0
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
            return 5  # default fallback
        return int(np.median(heights))

    def remove_foreign_object(self, image):
        # Improve foreign object removal by using bar thickness
        # First enhance contrast and maybe fix frequency domain before estimating thickness
        tmp = image.copy()
        tmp = self.fix_contrast(tmp)
        tmp_bar_thickness = self.estimate_bar_thickness(tmp)
        if tmp_bar_thickness < 3:
            tmp_bar_thickness = 3

        gray = image.copy()
        vertical_profile = np.mean(gray, axis=0)
        threshold = np.mean(vertical_profile) + 1.5 * np.std(vertical_profile)
        anomalous = vertical_profile > threshold

        cleaned = gray.copy()
        # Replace anomalous columns by taking neighbors within the bar thickness
        for i in range(gray.shape[1]):
            if anomalous[i]:
                # Look left and right up to bar_thickness columns
                left_idx = i-1
                while left_idx >= 0 and left_idx >= i - tmp_bar_thickness and anomalous[left_idx]:
                    left_idx -= 1
                right_idx = i+1
                while right_idx < gray.shape[1] and right_idx <= i + tmp_bar_thickness and anomalous[right_idx]:
                    right_idx += 1

                if left_idx >= 0 and left_idx >= i - tmp_bar_thickness and not (left_idx < 0 or anomalous[left_idx]):
                    cleaned[:, i] = gray[:, left_idx]
                elif right_idx < gray.shape[1] and right_idx <= i + tmp_bar_thickness and not anomalous[right_idx]:
                    cleaned[:, i] = gray[:, right_idx]

        # Also apply a horizontal pass if needed
        horizontal_profile = np.mean(cleaned, axis=1)
        h_threshold = np.mean(horizontal_profile) + 1.5 * np.std(horizontal_profile)
        h_anomalous = horizontal_profile > h_threshold
        for j in range(cleaned.shape[0]):
            if h_anomalous[j]:
                top = j-1
                while top >= 0 and top >= j - tmp_bar_thickness and h_anomalous[top]:
                    top -= 1
                bottom = j+1
                while bottom < cleaned.shape[0] and bottom <= j + tmp_bar_thickness and h_anomalous[bottom]:
                    bottom += 1

                if top >= 0 and top >= j - tmp_bar_thickness and not h_anomalous[top]:
                    cleaned[j, :] = cleaned[top, :]
                elif bottom < cleaned.shape[0] and bottom <= j + tmp_bar_thickness and not h_anomalous[bottom]:
                    cleaned[j, :] = cleaned[bottom, :]

        return cleaned

    def preprocess_image(self, image):
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        processed = gray.copy()

        # Fix order:
        # 1. Foreign object
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed)
            self.add_step("Removed Foreign Object", processed)

        # 2. Frequency domain
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed)
            self.add_step("Fixed Frequency Domain", processed)

        # 3. Salt & Pepper
        if 'salt_pepper' in problems:
            processed = self.remove_salt_and_pepper(processed)
            self.add_step("Removed Salt & Pepper", processed)

        # 4. Blur
        if 'blur' in problems:
            processed = self.fix_blur(processed)
            self.add_step("Fixed Blur", processed)

        # 5. Low Contrast
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
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

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

        for i in range(len(self.steps), rows*cols):
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
