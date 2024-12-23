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
        """Save each step for visualization and debugging."""
        self.steps.append((description, image.copy()))
    
    # ------------------------------------------------
    # 1. Bar Thickness Estimation
    # ------------------------------------------------
    def estimate_bar_thickness(self, image: np.ndarray) -> tuple:
        """
        Estimate minimal bar width and pattern width by analyzing vertical edges.
        Caps min_width at 3 and pattern_width at 9 for safer usage.
        """
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        # Focus on the middle portion for stable detection
        mid_slice = sobelx[sobelx.shape[0] // 4 : 3 * sobelx.shape[0] // 4]
        for row in mid_slice:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 1, 3  # Fallback if detection fails

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        return min(min_width, 3), min(pattern_width, 9)

    # ------------------------------------------------
    # 2. Barcode Region Detection
    # ------------------------------------------------
    def detect_barcode_region(self, image: np.ndarray):
        """
        Detect potential barcode region by looking for wide vertical edges 
        and filtering by aspect ratio + edge density.
        """
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        # Large blur + Otsu's threshold for region detection
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Extract contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Basic size checks
            if w < 20 or h < 10:
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)

            # Accept a broader aspect ratio range, e.g. up to 10
            if 1.5 < aspect_ratio < 10 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        # Return largest region by area
        return max(valid_regions, key=lambda r: r[2] * r[3])

    # ------------------------------------------------
    # 3. Problem Detection
    # ------------------------------------------------
    def detect_image_problems(self, image: np.ndarray):
        """
        Analyze the ROI for various issues: foreign_object, rotation, blur, 
        low_contrast, salt_pepper, frequency_domain.
        """
        problems = []
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) > 2 
            else image.copy()
        )

        # Attempt to detect region
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y : y + h, x : x + w]

        # Estimate bar thickness
        self.min_bar_width, self.pattern_width = self.estimate_bar_thickness(roi)

        # 3.1 Foreign object detection
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)
        if np.any(np.abs(vertical_profile - v_mean) > 2.5 * v_std):
            problems.append('foreign_object')

        # 3.2 Rotation detection
        edges = cv2.Canny(roi, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            try:
                angle = cv2.minAreaRect(coords)[-1]
                if abs(angle) > 5:
                    problems.append('rotation')
            except:
                pass

        # 3.3 Blur detection
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        if laplacian_var < 80:
            problems.append('blur')

        # 3.4 Low contrast
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 50:
            problems.append('low_contrast')

        # 3.5 Salt and pepper
        median_blurred = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median_blurred)
        if np.mean(noise_diff) > 10:
            problems.append('salt_pepper')

        # 3.6 Frequency domain
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        return problems, barcode_region

    # ------------------------------------------------
    # 4. Correction Methods
    # ------------------------------------------------

    # 4.1 Foreign Object Removal
    def remove_foreign_object(self, image: np.ndarray, barcode_region: tuple) -> np.ndarray:
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        # Analyze vertical profile
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)

        # Determine bar width to refine the averaging window
        min_bar_width, _ = self.estimate_bar_thickness(roi)
        min_bar_width = min(min_bar_width, 3)

        anomalies = np.abs(vertical_profile - v_mean) > 2.5 * v_std

        cleaned = roi.copy()
        for i in range(w):
            if anomalies[i]:
                # Instead of 1 col, average over [i - min_bar_width : i + min_bar_width]
                start = max(0, i - min_bar_width)
                end = min(w, i + min_bar_width)
                region = roi[:, start:end]
                avg_col = np.mean(region, axis=1)
                cleaned[:, i] = avg_col

        result = image.copy()
        result[y : y + h, x : x + w] = cleaned
        return result

    # 4.2 Salt & Pepper Removal
    def remove_salt_pepper(self, image: np.ndarray, barcode_region: tuple) -> np.ndarray:
        """
        Progressive median filtering to remove salt & pepper noise.
        """
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)

        noise_level = np.std(filtered - roi)
        if noise_level > 30:
            filtered = cv2.medianBlur(filtered, 3)

        result_img = image.copy()
        result_img[y : y + h, x : x + w] = filtered
        return result_img

    # 4.3 Rotation Fix
    def fix_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Use Hough lines to find rotation angle. Add padding 
        before warpAffine to prevent cropping.
        """
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

                # Add padding
                pad = int(max(rows, cols) * 0.3)
                padded = cv2.copyMakeBorder(
                    image, pad, pad, pad, pad, 
                    cv2.BORDER_CONSTANT, value=255
                )

                rows_p, cols_p = padded.shape
                # Build rotation matrix
                matrix = cv2.getRotationMatrix2D((cols_p / 2, rows_p / 2), median_angle, 1)
                rotated = cv2.warpAffine(
                    padded, matrix, (cols_p, rows_p),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )

                # Crop to content
                edges_rot = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges_rot > 0))
                if len(coords) > 0:
                    rx, ry, rw, rh = cv2.boundingRect(coords)
                    # ensure in-bounds
                    rx = max(0, rx)
                    ry = max(0, ry)
                    rw = min(rw, rotated.shape[1] - rx)
                    rh = min(rh, rotated.shape[0] - ry)
                    return rotated[ry : ry + rh, rx : rx + rw]

        return image

    # 4.4 Contrast Enhancement
    def fix_contrast(self, image: np.ndarray, barcode_region: tuple) -> np.ndarray:
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)

        result = image.copy()
        result[y : y + h, x : x + w] = enhanced
        return result

    # 4.5 Blur Fix
    def fix_blur(self, image: np.ndarray, barcode_region: tuple) -> np.ndarray:
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        # A slightly gentler approach
        unsharp = cv2.addWeighted(roi, 1.3, gaussian, -0.3, 0)

        result = image.copy()
        result[y : y + h, x : x + w] = unsharp
        return result

    # 4.6 Frequency Domain Fix
    def fix_frequency_domain(self, image: np.ndarray, barcode_region: tuple) -> np.ndarray:
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = roi.shape
        crow, ccol = rows // 2, cols // 2

        # Create a circular high-pass filter
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 30, 1, -1)
        # invert to get high-pass
        mask = 1 - mask

        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y : y + h, x : x + w] = cleaned
        return result

    # ------------------------------------------------
    # 5. Binarization + Cropping + Final Cleanup
    # ------------------------------------------------
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        A gentle approach to binarization. 
        Could also use adaptive threshold or Otsu's threshold.
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def find_optimal_crop(self, binary: np.ndarray) -> np.ndarray:
        """Crop to the main barcode region with additional padding for numbers."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Consider top few largest contours
        for contour in contours[:3]:
            x, y, w, h = cv2.boundingRect(contour)
            # If contour is large enough, assume it's the barcode
            if w > binary.shape[1] * 0.1 and h > binary.shape[0] * 0.1:
                # Extra padding
                pad_x = int(0.05 * w)
                pad_y = int(0.2 * h)  # More padding below for numbers

                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(binary.shape[1] - x, w + 2 * pad_x)
                h = min(binary.shape[0] - y, h + pad_y + int(0.1 * h))

                cropped = binary[y : y + h, x : x + w]

                # If not empty
                if np.sum(cropped) > 0:
                    return cropped

        return binary

    def final_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Perform a final vertical averaging filter to unify leftover anomalies.
        """
        if binary is None or np.sum(binary) == 0:
            return binary

        h, w = binary.shape
        min_bar_width, _ = self.estimate_bar_thickness(binary)
        min_bar_width = max(1, min(min_bar_width, 3))

        # Create a vertical kernel
        kernel = np.ones((h, min_bar_width)) / h

        cleaned = np.zeros_like(binary)
        for i in range(w - min_bar_width + 1):
            region = binary[:, i : i + min_bar_width]
            avg_val = np.mean(region)
            cleaned[:, i] = (avg_val > 127) * 255

        return cleaned

    # ------------------------------------------------
    # 6. Main Processing Pipeline
    # ------------------------------------------------
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Main pipeline applying all fixes in optimal order."""
        self.steps = []
        self.add_step("Original Image", image)

        # Convert to grayscale
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) > 2 
            else image.copy()
        )
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        # If no region found, skip
        if barcode_region is None:
            return gray

        # Extract initial region
        x, y, w, h = barcode_region
        processed = gray.copy()

        # 1. Rotation
        if 'rotation' in problems:
            processed = self.fix_rotation(processed)
            self.add_step("Fixed Rotation", processed)

            # Re-detect region after rotation
            new_region = self.detect_barcode_region(processed)
            if new_region:
                x, y, w, h = new_region
                processed = processed[y : y + h, x : x + w]
            else:
                # fallback to original region
                processed = processed[y : y + h, x : x + w]
        else:
            processed = processed[y : y + h, x : x + w]

        # 2. Foreign Object
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(
                processed, (0, 0, processed.shape[1], processed.shape[0])
            )
            self.add_step("Removed Foreign Object", processed)

        # 3. Frequency Domain
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(
                processed, (0, 0, processed.shape[1], processed.shape[0])
            )
            self.add_step("Fixed Frequency Domain", processed)

        # 4. Salt & Pepper
        if 'salt_pepper' in problems:
            processed = self.remove_salt_pepper(
                processed, (0, 0, processed.shape[1], processed.shape[0])
            )
            self.add_step("Removed Salt & Pepper", processed)

        # 5. Blur
        if 'blur' in problems:
            processed = self.fix_blur(
                processed, (0, 0, processed.shape[1], processed.shape[0])
            )
            self.add_step("Fixed Blur", processed)

        # 6. Low Contrast
        if 'low_contrast' in problems:
            processed = self.fix_contrast(
                processed, (0, 0, processed.shape[1], processed.shape[0])
            )
            self.add_step("Enhanced Contrast", processed)

        # 7. Binarize
        binary = self.binarize(processed)
        self.add_step("Binarized", binary)

        # 8. Crop
        cropped = self.find_optimal_crop(binary)
        self.add_step("Cropped", cropped)

        # 9. Final Cleanup
        final = self.final_cleanup(cropped)
        self.add_step("Final Cleanup", final)

        return final

    # ------------------------------------------------
    # 7. Visualization
    # ------------------------------------------------
    def visualize_steps(self):
        """Show each step in a grid for debugging and analysis."""
        n_steps = len(self.steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        if rows * cols == 1:
            # Single-subplot edge case
            axes = np.array([axes])

        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for i, (desc, img) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(desc, fontsize=8)
            ax.axis('off')

        # Hide any unused subplots
        for i in range(len(self.steps), rows * cols):
            r = i // cols
            c = i % cols
            axes[r, c].axis('off')

        plt.tight_layout()
        plt.show()


# ------------------------------------------------
# Main Execution
# ------------------------------------------------
if __name__ == "__main__":
    preprocessor = BarcodePreprocessor()
    image_path = filedialog.askopenfilename(
        title="Select an Image", 
    )
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()
        else:
            print("Failed to load image.")
    else:
        print("No image selected.")
