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
        """Estimate barcode bar thickness using vertical edge analysis"""
        # Get vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)
        
        # Find typical bar width using peaks
        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:  # Use middle portion
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))
        
        if not widths:
            return 3  # Default fallback
        
        min_width = max(1, int(np.percentile(widths, 10)))
        return min_width

    def detect_barcode_region(self, image):
        """Find barcode region using vertical edges"""
        # Edge enhancement
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        blur = cv2.GaussianBlur(gradx, (21,21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Filter valid regions
        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:  # Too small
                continue
            aspect = w/h
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50)/(w*h)
            if 1.5 < aspect < 5 and edge_density > 0.1:
                valid_regions.append((x,y,w,h))
                
        if not valid_regions:
            return None
            
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def detect_image_problems(self, image):
        """Simplified problem detection with reduced false positives"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Find barcode region
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return [], (0,0,gray.shape[1],gray.shape[0])
            
        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]
        
        # Blur detection (from codebase 1)
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        if laplacian_var < 100:  # Original threshold that worked
            problems.append('blur')
            
        # Low contrast detection
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 50:  # Original threshold from codebase 1
            problems.append('low_contrast')
            
        # Salt and pepper (original working detection)
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.mean(noise_diff) > 10:
            problems.append('salt_pepper')
            
        # Frequency domain detection
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20*np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')
            
        # Foreign object detection using bar thickness
        bar_width = self.estimate_bar_thickness(roi)
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        anomalies = np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0 * v_std
        if np.sum(anomalies) > bar_width * 2:
            problems.append('foreign_object')
            
        # Rotation detection (from codebase 1)
        edges = cv2.Canny(roi, 100, 200)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')
                
        return problems, barcode_region

    def fix_rotation(self, image, barcode_region):
        """Rotation fix from codebase 1"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
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
                rows, cols = roi.shape
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                
                # Add padding to avoid cropping
                pad = int(max(w,h) * 0.2)
                padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, 
                                          cv2.BORDER_CONSTANT, value=255)
                
                rotated = cv2.warpAffine(padded, matrix, (padded.shape[1], padded.shape[0]),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
                
                # Find barcode in rotated image
                edges = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges > 0))
                x_, y_, w_, h_ = cv2.boundingRect(coords)
                
                rotated_crop = rotated[y_:y_+h_, x_:x_+w_]
                if rotated_crop.shape != (h,w):
                    rotated_crop = cv2.resize(rotated_crop, (w,h))
                    
                result = image.copy()
                result[y:y+h, x:x+w] = rotated_crop
                return result
                
        return image

    def remove_foreign_object(self, image, barcode_region):
        """Foreign object removal using vertical kernel filtering"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        # Get bar width for kernel size
        bar_width = self.estimate_bar_thickness(roi)
        
        # Create vertical averaging kernel
        kernel_height = h
        kernel_width = bar_width
        kernel = np.ones((kernel_height, kernel_width)) / kernel_height
        
        # Apply kernel to detect and fix anomalies
        cleaned = roi.copy()
        for i in range(w - kernel_width + 1):
            region = roi[:, i:i+kernel_width]
            avg_val = np.mean(region, axis=0)
            std_val = np.std(region, axis=0)
            
            # If significant variation, apply vertical averaging
            if np.any(std_val > np.mean(std_val) * 2):
                cleaned[:, i:i+kernel_width] = cv2.filter2D(region, -1, kernel)
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_frequency_domain(self, image, barcode_region):
        """Frequency domain fix from codebase 3"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        mask_area = np.sqrt((np.arange(rows)[:,None] - crow)**2 + 
                          (np.arange(cols)[None,:] - ccol)**2) <= r
        mask[mask_area] = 0
        
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_blur(self, image, barcode_region):
        """Single-stage conservative unsharp masking"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        gaussian = cv2.GaussianBlur(roi, (0,0), 3)
        unsharp = cv2.addWeighted(roi, 1.5, gaussian, -0.5, 0)
        
        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def fix_contrast(self, image, barcode_region):
        """Conservative contrast enhancement"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        
        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def remove_salt_pepper(self, image, barcode_region):
        """Salt and pepper removal from codebase 1"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        
        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)
        
        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def binarize(self, image):
        """Conservative binarization to prevent holes"""
        # Normalize
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            normalized, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Larger window
            7
        )
        
        # Light cleanup
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def final_cleanup(self, binary):
        """Final cleanup using vertical averaging"""
        bar_width = self.estimate_bar_thickness(binary)
        
        kernel = np.ones((binary.shape[0], bar_width)) / binary.shape[0]
        cleaned = np.zeros_like(binary)
        
        for i in range(binary.shape[1] - bar_width + 1):
            region = binary[:, i:i+bar_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i+bar_width] = (vertical_avg > 127).astype(np.uint8) * 255
            
        return cleaned

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Detect problems
        problems, barcode_region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)
        
        if barcode_region is None:
            return gray
            
        processed = gray.copy()
        x, y, w, h = barcode_region
        
        # Process in optimal order
        # 1. Fix rotation first
        if 'rotation' in problems:
            processed = self.fix_rotation(processed, barcode_region)
            self.add_step("Fixed Rotation", processed)
            
            # Update region after rotation
            new_region = self.detect_barcode_region(processed)
            if new_region:
                barcode_region = new_region
                x, y, w, h = barcode_region
        
        # 2. Fix frequency domain issues
        if 'frequency_domain' in problems:
            processed = self.fix_frequency_domain(processed, barcode_region)
            self.add_step("Fixed Frequency Domain", processed)
        
        # 3. Remove foreign objects
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed, barcode_region)
            self.add_step("Removed Foreign Object", processed)
        
        # 4. Fix salt & pepper noise
        if 'salt_pepper' in problems:
            processed = self.remove_salt_pepper(processed, barcode_region)
            self.add_step("Removed Salt & Pepper", processed)
        
        # 5. Fix blur
        if 'blur' in problems:
            processed = self.fix_blur(processed, barcode_region)
            self.add_step("Fixed Blur", processed)
        
        # 6. Fix contrast
        if 'low_contrast' in problems:
            processed = self.fix_contrast(processed, barcode_region)
            self.add_step("Enhanced Contrast", processed)
        
        # 7. Extract ROI after all fixes
        roi = processed[y:y+h, x:x+w]
        
        # 8. Binarize
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)
        
        # 9. Final cleanup using vertical averaging
        cleaned = self.final_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)
        
        return cleaned
        
    def visualize_steps(self):
        """Enhanced visualization with proper subplot handling"""
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
            processed = preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()