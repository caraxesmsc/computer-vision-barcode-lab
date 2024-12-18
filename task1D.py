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

    # -------------------------------------
    # UTILITY FUNCTIONS
    # -------------------------------------
    def estimate_bar_thickness(self, image):
        """Estimate barcode bar thickness for pattern-aware processing"""
        # Initial preprocessing to improve estimation
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

        # Get vertical gradient
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        # Find typical bar width using middle portion of image
        widths = []
        for row in sobelx[sobelx.shape[0] // 4:3 * sobelx.shape[0] // 4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3, 9  # Default fallback

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))

        return min_width, pattern_width

    def get_anomalous_regions(self, anomalies, bar_thickness):
        """Group contiguous anomalous regions based on bar thickness"""
        regions = []
        if len(anomalies) > 0:
            current_region = [anomalies[0]]
            for i in range(1, len(anomalies)):
                if anomalies[i] == anomalies[i - 1] + 1 and anomalies[i] - current_region[0] < bar_thickness * 2:
                    current_region.append(anomalies[i])
                else:
                    regions.append(current_region)
                    current_region = [anomalies[i]]
            regions.append(current_region)
        return regions

    def validate_barcode_region(self, region, image):
        """Validate that a region is likely to be a barcode region"""
        x, y, w, h = region
        roi = image[y:y + h, x:x + w]

        # Check aspect ratio
        aspect = w / h
        if not (1.5 < aspect < 5):
            return False

        # Verify vertical line pattern
        edges = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        edge_density = np.mean(np.abs(edges) > 50)
        return edge_density > 0.1
    
    def detect_barcode_region(self, image):
        """Find the actual barcode region in the image"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        # Find regions with high concentration of vertical lines
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
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
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)

            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        # Return region with highest edge density
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]))

    # -------------------------
    # PROBLEM DETECTION
    # -------------------------
    def detect_image_problems(self, image):
        """Detect image problems"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Find barcode region for focused analysis
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])
        
        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]
        
        # Estimate bar width and pattern width
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)

        # --- Foreign object detection ---
        if self.detect_foreign_object(roi, min_bar_width, pattern_width):
            problems.append('foreign_object')

        # --- Blur detection ---
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.mean(edges > 0)
        if laplacian_var < 90 and edge_density < 0.15:
            problems.append('blur')

        # --- Low contrast detection ---
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')

        # --- Salt and pepper detection ---
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.percentile(noise_diff, 95) > 50:
            problems.append('salt_pepper')
            
        # --- Compression artifacts detection ---
        if self.detect_compression_artifacts(roi):
            problems.append('compression_artifacts')

        # --- Frequency domain detection ---
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        # --- Rotation detection ---
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(w,h)//3)
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

    def detect_foreign_object(self, roi, min_bar_width, pattern_width):
        """Detect foreign objects using pattern analysis"""
        vertical_profile = np.mean(roi, axis=0)
        
        # Use rolling window to detect anomalous regions
        window_size = pattern_width * 2
        v_std = np.std([vertical_profile[i:i+window_size].std() 
                        for i in range(len(vertical_profile)-window_size)])
        
        # Look for sections where local variance deviates significantly from pattern
        return v_std > 15 

    def detect_compression_artifacts(self, image):
        """Detect compression artifacts using edge analysis"""
        # Apply Sobel filters
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Look for characteristic blocky patterns
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        blocks = cv2.boxFilter(magnitude, -1, (8,8))

        # Check for block boundary artifacts
        block_variation = np.std(blocks)

        return block_variation > 10  # Threshold
    
    def remove_salt_and_pepper(self, image, barcode_region):
        """Remove salt and pepper noise using median filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Progressive median filtering
        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)

        # Check noise level
        noise_level = np.std(filtered - roi)
        if noise_level > 20:
            filtered = cv2.medianBlur(filtered, 3)

        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    # -------------------------
    # FIXING FUNCTIONS
    # -------------------------
    def remove_foreign_object(self, image, barcode_region):
        """Remove foreign objects using pattern-aware interpolation"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)
        
        
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)

        
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)

       
        v_threshold = np.mean(vertical_profile) + 2.5 * v_std
        h_threshold = np.mean(horizontal_profile) + 2.5 * h_std

        
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > v_threshold
        h_anomalous = np.abs(horizontal_profile - np.mean(horizontal_profile)) > h_threshold
        
        
        
        v_anomalous_regions = self.get_anomalous_regions(np.where(v_anomalous)[0], min_bar_width)
        h_anomalous_regions = self.get_anomalous_regions(np.where(h_anomalous)[0], min_bar_width)

        cleaned = roi.copy()

        
        for region in v_anomalous_regions:
            for i in region:
                
                
                left_idx = i - 1
                while left_idx >= 0 and left_idx >= i - pattern_width and v_anomalous[left_idx]:
                    left_idx -= 1

                
                right_idx = i + 1
                while right_idx < w and right_idx <= i + pattern_width and v_anomalous[right_idx]:
                    right_idx += 1

                
                if left_idx >= 0 and left_idx >= i - pattern_width and not v_anomalous[left_idx]:
                    
                    
                    cleaned[:, i] = roi[:, left_idx]
                elif right_idx < w and right_idx <= i + pattern_width and not v_anomalous[right_idx]:
                    
                    
                    cleaned[:, i] = roi[:, right_idx]

       
        for region in h_anomalous_regions:
            for j in region:
                
                top_idx = j - 1
                while top_idx >= 0 and top_idx >= j - pattern_width and h_anomalous[top_idx]:
                    top_idx -= 1

                
                bottom_idx = j + 1
                while bottom_idx < h and bottom_idx <= j + pattern_width and h_anomalous[bottom_idx]:
                    bottom_idx += 1

               
                if top_idx >= 0 and top_idx >= j - pattern_width and not h_anomalous[top_idx]:
                    
                    
                    cleaned[j, :] = roi[top_idx,:]
                elif bottom_idx < h and bottom_idx <= j + pattern_width and not h_anomalous[bottom_idx]:
                    
                   
                    cleaned[j, :] = roi[bottom_idx,:]
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Enhanced frequency domain correction using Gaussian high-pass filter"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)

        # Create a Gaussian high-pass filter
        rows, cols = roi.shape
        crow, ccol = rows // 2, cols // 2
        x = np.linspace(-ccol, ccol - 1, cols)
        y = np.linspace(-crow, crow - 1, rows)
        xv, yv = np.meshgrid(x, y)
        sigma = 30  # Adjust sigma for filter strength
        gaussian_filter = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        mask = 1 - gaussian_filter  # Invert to make it high-pass

        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y:y + h, x:x + w] = cleaned
        return result

    def fix_blur(self, image, barcode_region):
        """Two-stage deblurring: unsharp masking and sharpening kernel"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.5, gaussian, -0.5, 0)

        # Second stage: Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        # Check if further sharpening is needed
        laplacian_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if laplacian_var < 150:  # Only sharpen more if still somewhat blurry
            sharpened = cv2.filter2D(unsharp, -1, kernel)
        else:
            sharpened = unsharp

        result = image.copy()
        result[y:y+h, x:x+w] = sharpened
        return result

    def fix_contrast(self, image, barcode_region, contrast_type=None):
        """Targeted contrast enhancement based on issue type"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        if contrast_type == 'dark_contrast':
            # For dark images: aggressive brightening while preserving blacks
            p2, p98 = np.percentile(roi, (2, 98))
            enhanced = np.interp(roi, (p2, p98), (0, 255)).astype(np.uint8)
            enhanced[roi < p2] = 0
        elif contrast_type == 'bright_contrast':
            # For bright images: increase black/white separation
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(roi)
            enhanced[roi > 250] = 255
        else:
            # General contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(roi)
            # Gamma correction 
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)

        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result
    
    def fix_compression_artifacts(self, image, barcode_region):
        """Apply mild noise reduction to fix compression artifacts"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        denoised = cv2.fastNlMeansDenoising(
            roi,
            None,
            h=10,  # Filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        result = image.copy()
        result[y:y+h, x:x+w] = denoised
        return result

    def fix_rotation(self, image, barcode_region):
        """Rotation correction with padding to avoid cropping barcode parts"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Add padding before rotation
        pad = int(max(w, h) * 0.2)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

        edges = cv2.Canny(padded, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(w,h)//3)

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
                center = (padded.shape[1] // 2, padded.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(padded, matrix, (padded.shape[1], padded.shape[0]),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)

                # Find the actual barcode region in the rotated image
                edges = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges > 0))
                if len(coords):
                    x_, y_, w_, h_ = cv2.boundingRect(coords)
                    rotated = rotated[y_:y_ + h_, x_:x_ + w_]

                    result = image.copy()
                    result[y:y+h, x:x+w] = rotated
                    return result

        return image

    def binarize(self, image):
        """Binarize the image using adaptive thresholding with pre-normalization"""
        # Normalize contrast
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)

        # Apply adaptive thresholding with carefully chosen parameters
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Slightly smaller window to prevent holes
            5    # Adjusted offset
        )

        # Light cleanup without creating holes
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)

        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        # Detect problems and barcode region
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

        if 'compression_artifacts' in problems:
            processed = self.fix_compression_artifacts(processed, barcode_region)
            self.add_step("Fixed Compression Artifacts", processed)

        if 'salt_pepper' in problems:
            processed = self.remove_salt_and_pepper(processed, barcode_region)
            self.add_step("Removed Salt & Pepper", processed)

        if 'blur' in problems:
            processed = self.fix_blur(processed, barcode_region)
            self.add_step("Fixed Blur", processed)

        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed, barcode_region)
            self.add_step("Enhanced Contrast", processed)
            
        # Binarize after all other fixes
        x, y, w, h = barcode_region
        binary = self.binarize(processed[y:y+h, x:x+w])
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

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
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