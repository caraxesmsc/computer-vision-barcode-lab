import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.min_bar_width = None
        self.pattern_width = None

    def add_step(self, description, image):
        self.steps.append((description, image.copy()))

    def estimate_bar_thickness(self, image):
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 1, 3

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        
        return min(min_width, 5), min(pattern_width, 15)  # Smaller caps

    def detect_barcode_region(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 10:
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        largest = max(valid_regions, key=lambda r: r[2] * r[3])
        return largest

    def detect_image_problems(self, image):
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]

        self.min_bar_width, self.pattern_width = self.estimate_bar_thickness(roi)

        # Foreign object detection
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)
        if np.any(np.abs(vertical_profile - v_mean) > 2.5 * v_std):
            problems.append('foreign_object')

        # Blur detection
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        if laplacian_var < 80:
            problems.append('blur')

        # Low contrast
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 50:
            problems.append('low_contrast')

        # Salt and pepper
        median_blurred = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median_blurred)
        if np.percentile(noise_diff, 95) > 30:
            problems.append('salt_pepper')

        # Rotation
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
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

        return problems, barcode_region

    def remove_foreign_object(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Apply 1-pixel vertical averaging
        cleaned = np.zeros_like(roi)
        for i in range(w):
            col = roi[:, i]
            cleaned[:, i] = np.mean(col) * np.ones(h)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def remove_salt_pepper(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Try different kernel combinations
        candidates = []
        
        # Single pass filters
        candidates.append(cv2.medianBlur(roi, 3))
        candidates.append(cv2.medianBlur(roi, 5))
        
        # Double pass combinations
        temp = cv2.medianBlur(roi, 3)
        candidates.append(cv2.medianBlur(temp, 5))
        
        temp = cv2.medianBlur(roi, 5)
        candidates.append(cv2.medianBlur(temp, 3))

        # Find best result based on edge preservation
        best_score = -float('inf')
        best_result = candidates[0]
        
        for candidate in candidates:
            edges = cv2.Canny(candidate, 50, 150)
            score = np.sum(edges)
            if score > best_score:
                best_score = score
                best_result = candidate

        result = image.copy()
        result[y:y+h, x:x+w] = best_result
        return result

    def fix_rotation(self, image):
        edges = cv2.Canny(image, 50, 150)
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
                
                # Add padding before rotation
                pad = int(max(rows, cols) * 0.3)
                padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                          cv2.BORDER_CONSTANT, value=255)
                
                # Calculate new dimensions
                angle_rad = abs(np.radians(median_angle))
                new_w = int(cols * abs(np.cos(angle_rad)) + rows * abs(np.sin(angle_rad)))
                new_h = int(rows * abs(np.cos(angle_rad)) + cols * abs(np.sin(angle_rad)))

                # Adjust matrix for new dimensions
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                matrix[0, 2] += (new_w - cols) / 2
                matrix[1, 2] += (new_h - rows) / 2

                rotated = cv2.warpAffine(padded, matrix, (new_w, new_h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)

                # Find content in rotated image
                edges = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges > 0))
                if len(coords):
                    x, y, w, h = cv2.boundingRect(coords)
                    return rotated[y:y+h, x:x+w]
        return image

    def fix_contrast(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        
        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def fix_blur(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.2, gaussian, -0.2, 0)  # Gentler parameters

        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def binarize(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        binary = cv2.adaptiveThreshold(
            normalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,  # Smaller window
            5
        )
        
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def final_cleanup(self, binary):
        try:
            min_width = max(1, self.min_bar_width)
        except:
            min_width = 1

        kernel = np.ones((binary.shape[0], min_width)) / binary.shape[0]
        cleaned = np.zeros_like(binary)

        for i in range(binary.shape[1] - min_width + 1):
            col_sum = np.sum(binary[:, i:i+min_width], axis=1) / min_width
            cleaned[:, i] = (col_sum > 127).astype(np.uint8) * 255

        return cleaned

    def find_optimal_crop(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add generous padding for numbers
        pad_y_top = int(0.05 * h)
        pad_y_bottom = int(0.2 * h)  # More padding below for numbers
        pad_x = int(0.05 * w)

        x = max(0, x - pad_x)
        y = max(0, y - pad_y_top)
        w = min(binary.shape[1] - x, w + 2*pad_x)
        h = min(binary.shape[0] - y, h + pad_y_top + pad_y_bottom)

        return binary[y:y+h, x:x+w]

    def preprocess_image(self, image):
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if barcode_region is None:
            return gray

        processed = gray.copy()
        x, y, w, h = barcode_region

        # Fix rotation first if needed
        if 'rotation' in problems:
            processed = self.fix_rotation(processed)
            self.add_step("Fixed Rotation", processed)
            
            # Re-detect barcode region
            new_region = self.detect_barcode_region(processed)
            if new_region:
                barcode_region = new_region
                x, y, w, h = barcode_region
            else:
                processed = gray.copy()
                barcode_region = self.detect_barcode_region(processed)
                if barcode_region:
                    x, y, w, h = barcode_region

        # Extract ROI
        roi = processed[y:y+h, x:x+w].copy()

        # Remove foreign objects first
        if 'foreign_object' in problems:
            roi = self.remove_foreign_object(roi, (0, 0, w, h))
            self.add_step("Removed Foreign Object", roi)

        # Fix salt & pepper next
        if 'salt_pepper' in problems:
            roi = self.remove_salt_pepper(roi, (0, 0, w, h))
            self.add_step("Removed Salt & Pepper", roi)

        # Fix blur
        if 'blur' in problems:
            roi = self.fix_blur(roi, (0, 0, w, h))
            self.add_step("Fixed Blur", roi)

        # Fix contrast last before binarization
        if 'low_contrast' in problems:
            roi = self.fix_contrast(roi, (0, 0, w, h))
            self.add_step("Enhanced Contrast", roi)

        # Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        # Final cleanup using vertical averaging
        cleaned = self.final_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        # Find optimal crop
        final = self.find_optimal_crop(cleaned)
        self.add_step("Final Result", final)

        return final

    def visualize_steps(self):
        n_steps = len(self.steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Handle single image case
        if rows * cols == 1:
            axes = np.array([axes])
            
        # Reshape axes for proper indexing
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
        else:
            print("Failed to load image.")