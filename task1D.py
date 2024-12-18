import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        
    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def estimate_bar_thickness(self, image):
        """Estimate barcode bar thickness for pattern-aware processing"""
        # Initial preprocessing to improve estimation
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3,3), 0)
        
        # Get vertical gradient
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)
        
        # Find typical bar width using middle portion of image
        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))
                
        if not widths:
            return 3  # Default fallback
            
        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        
        return min_width, pattern_width

    def detect_barcode_region(self, image):
        """Find the actual barcode region in the image"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        # Find regions with high concentration of vertical lines
        blur = cv2.GaussianBlur(gradx, (21,21), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
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
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50) / (w * h)
            
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))
        
        if not valid_regions:
            return None
            
        # Return region with highest edge density
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def detect_image_problems(self, image):
        """Enhanced problem detection combining best elements from all versions"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Get barcode region for focused analysis
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])
            
        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]

        # Get pattern characteristics
        min_width, pattern_width = self.estimate_bar_thickness(roi)
        
        # Foreign object detection using pattern analysis
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        
        # Look for anomalies considering pattern width
        v_threshold = np.mean(vertical_profile) + 2.5 * v_std
        h_threshold = np.mean(horizontal_profile) + 2.5 * h_std
        
        v_anomalies = np.abs(vertical_profile - np.mean(vertical_profile)) > v_threshold
        h_anomalies = np.abs(horizontal_profile - np.mean(horizontal_profile)) > h_threshold
        
        if np.sum(v_anomalies) > min_width or np.sum(h_anomalies) > min_width:
            problems.append('foreign_object')

        # Blur detection using gradient analysis
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.mean(edges > 0)
        
        if laplacian_var < 100 and edge_density < 0.15:
            problems.append('blur')

        # Contrast detection
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')

        # Salt and pepper detection
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.percentile(noise_diff, 95) > 30:
            problems.append('salt_pepper')

        # Frequency domain analysis
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Check high frequency components
        high_freq_energy = np.mean(magnitude_spectrum[0:magnitude_spectrum.shape[0]//4, :]) + \
                          np.mean(magnitude_spectrum[-magnitude_spectrum.shape[0]//4:, :])
        
        if high_freq_energy > 45:
            problems.append('frequency_domain')

        # Compression artifacts detection
        if np.std(magnitude_spectrum) > 45:
            problems.append('compression_artifacts')

        # Rotation detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(w, h)//3)
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
        """Remove foreign objects using pattern-aware interpolation"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        min_width, pattern_width = self.estimate_bar_thickness(roi)
        
        # Get profiles and detect anomalies
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        
        v_threshold = np.mean(vertical_profile) + 2.5 * v_std
        h_threshold = np.mean(horizontal_profile) + 2.5 * h_std
        
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > v_threshold
        h_anomalous = np.abs(horizontal_profile - np.mean(horizontal_profile)) > h_threshold
        
        cleaned = roi.copy()
        
        # Process vertical anomalies
        for i in range(w):
            if v_anomalous[i]:
                left_vals = []
                right_vals = []
                
                # Look for valid pattern within pattern_width distance
                for j in range(max(0, i-pattern_width), i):
                    if not v_anomalous[j]:
                        left_vals.append(roi[:, j])
                
                for j in range(i+1, min(w, i+pattern_width+1)):
                    if not v_anomalous[j]:
                        right_vals.append(roi[:, j])
                
                if left_vals and right_vals:
                    # Use average of nearest valid patterns
                    cleaned[:, i] = np.mean([left_vals[-1], right_vals[0]], axis=0)
                elif left_vals:
                    cleaned[:, i] = left_vals[-1]
                elif right_vals:
                    cleaned[:, i] = right_vals[0]
        
        # Process horizontal anomalies
        for i in range(h):
            if h_anomalous[i]:
                top_vals = []
                bottom_vals = []
                
                for j in range(max(0, i-pattern_width), i):
                    if not h_anomalous[j]:
                        top_vals.append(cleaned[j, :])
                
                for j in range(i+1, min(h, i+pattern_width+1)):
                    if not h_anomalous[j]:
                        bottom_vals.append(cleaned[j, :])
                
                if top_vals and bottom_vals:
                    cleaned[i, :] = np.mean([top_vals[-1], bottom_vals[0]], axis=0)
                elif top_vals:
                    cleaned[i, :] = top_vals[-1]
                elif bottom_vals:
                    cleaned[i, :] = bottom_vals[0]
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Enhanced frequency domain correction"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create butterworth high-pass filter
        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2
        
        u = np.arange(rows)
        v = np.arange(cols)
        u = u.reshape(rows,1)
        u = np.repeat(u, cols, axis=1)
        v = np.repeat(v.reshape(1,cols), rows, axis=0)
        
        D = np.sqrt((u-crow)**2 + (v-ccol)**2)
        D = np.where(D == 0, 1, D)  # Avoid division by zero
        n = 2  # Filter order
        D0 = 30  # Cutoff frequency
        
        H = 1 / (1 + (D0/D)**(2*n))
        
        f_shift = f_shift * H
        
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize back to uint8
        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_blur(self, image, barcode_region):
        """Two-stage deblurring"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.5, gaussian, -0.5, 0)
        
        # Check if more sharpening needed
        laplacian_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if laplacian_var < 150:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            unsharp = cv2.filter2D(unsharp, -1, kernel)
        
        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def fix_contrast(self, image, barcode_region):
        """Multi-stage contrast enhancement"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # First stage: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        
        # Second stage: Adaptive gamma correction
        mean_intensity = np.mean(enhanced)
        if mean_intensity < 128:
            # Dark image: gamma > 1
            gamma = 1.5
        else:
            # Bright image: gamma < 1
            gamma = 0.7
            
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def remove_salt_pepper(self, image, barcode_region):
        """Enhanced salt and pepper removal"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Progressive median filtering
        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)
        
        # Check noise level
        noise_level = np.std(filtered - roi)
        if noise_level > 30:
            filtered = cv2.medianBlur(filtered, 3)
        
        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def fix_rotation(self, image, barcode_region):
        """Rotation correction with proper padding"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(w,h)//3)
        
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
                
                # Calculate new dimensions post-rotation
                angle_rad = np.radians(abs(median_angle))
                new_w = int(w * abs(np.cos(angle_rad)) + h * abs(np.sin(angle_rad)))
                new_h = int(h * abs(np.cos(angle_rad)) + w * abs(np.sin(angle_rad)))
                
                # Add padding before rotation
                pad_x = (new_w - w) // 2
                pad_y = (new_h - h) // 2
                
                padded = cv2.copyMakeBorder(roi, pad_y, pad_y, pad_x, pad_x,
                                          cv2.BORDER_CONSTANT, value=255)
                
                # Perform rotation
                center = (padded.shape[1] // 2, padded.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(padded, matrix, (padded.shape[1], padded.shape[0]),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
                
                # Find the actual barcode region in rotated image
                edges = cv2.Canny(rotated, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour with appropriate aspect ratio
                    valid_contours = []
                    for contour in contours:
                        x_, y_, w_, h_ = cv2.boundingRect(contour)
                        aspect_ratio = w_ / float(h_)
                        if 1.5 < aspect_ratio < 5:
                            valid_contours.append((x_, y_, w_, h_))
                    
                    if valid_contours:
                        x_, y_, w_, h_ = max(valid_contours, 
                                           key=lambda b: b[2] * b[3])
                        return rotated[y_:y_+h_, x_:x_+w_]
        
        return roi

    def binarize(self, image):
        """Improved binarization with hole prevention"""
        # Normalize contrast first
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        # Apply adaptive thresholding with carefully chosen parameters
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Slightly smaller window to prevent holes
            7    # Adjusted offset
        )
        
        # Light cleanup without creating holes
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Detect problems and get barcode region
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)
        
        if barcode_region is None:
            return gray
        
        processed = gray.copy()
        
        # Apply fixes in optimal order
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed, barcode_region)
            self.add_step("Removed Foreign Object", processed)
        
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed, barcode_region)
            self.add_step("Fixed Frequency Domain", processed)
        
        if 'salt_pepper' in problems:
            processed = self.remove_salt_pepper(processed, barcode_region)
            self.add_step("Removed Salt & Pepper", processed)
        
        if 'blur' in problems:
            processed = self.fix_blur(processed, barcode_region)
            self.add_step("Fixed Blur", processed)
        
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed, barcode_region)
            self.add_step("Enhanced Contrast", processed)
        
        # Get updated barcode region after fixes
        _, updated_region = self.detect_image_problems(processed)
        if updated_region is None:
            updated_region = barcode_region
            
        # Extract ROI for final processing
        x, y, w, h = updated_region
        roi = processed[y:y+h, x:x+w]
        
        # Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)
        
        # Fix rotation if needed (do this after binarization)
        if 'rotation' in problems:
            binary = self.fix_rotation(binary, (0, 0, binary.shape[1], binary.shape[0]))
            self.add_step("Fixed Rotation", binary)
        
        return binary

    def visualize_steps(self):
        """Enhanced visualization with proper subplot handling"""
        n_steps = len(self.steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
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