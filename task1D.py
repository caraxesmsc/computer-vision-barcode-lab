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

    def estimate_barcode_pattern(self, image):
        """Analyze barcode pattern to get bar width and spacing"""
        # Initial preprocessing to make pattern detection easier
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3,3), 0)
        
        # Get vertical gradient
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)
        
        # Find typical bar width
        widths = []
        for row in sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]:  # Use middle portion
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))
        
        if not widths:
            return 3, 9  # Default fallback
        
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
            
        # Filter contours by aspect ratio and density of vertical lines
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
            
        # Return the region with highest edge density
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def detect_image_problems(self, image, barcode_region=None):
        """More precise problem detection"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        roi = gray
        if barcode_region is not None:
            x, y, w, h = barcode_region
            roi = gray[y:y+h, x:x+w]
        
        # Get barcode pattern info
        min_bar_width, pattern_width = self.estimate_barcode_pattern(roi)
        
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
        
        # Salt and pepper noise detection
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.percentile(noise_diff, 95) > 30:
            problems.append('salt_pepper')
        
        # Foreign object detection
        avg_intensity = np.mean(roi, axis=0)
        smoothed = np.convolve(avg_intensity, np.ones(pattern_width)/pattern_width, mode='same')
        anomalies = np.abs(avg_intensity - smoothed) > 2 * np.std(avg_intensity)
        if np.sum(anomalies) > min_bar_width:
            problems.append('foreign_object')
        
        # Rotation detection
        edges = cv2.Canny(roi, 50, 150)
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

    def remove_foreign_object(self, image, barcode_region):
        """Remove foreign objects using barcode pattern"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w]
        
        min_bar_width, pattern_width = self.estimate_barcode_pattern(roi)
        vertical_avg = np.mean(roi, axis=0)
        
        # Find anomalous regions
        smoothed = np.convolve(vertical_avg, np.ones(pattern_width)/pattern_width, mode='same')
        anomalies = np.abs(vertical_avg - smoothed) > 2 * np.std(vertical_avg)
        
        # Process each anomalous region
        cleaned = roi.copy()
        for i in range(len(anomalies)):
            if anomalies[i]:
                # Look for good regions within pattern_width
                left_idx = max(0, i - pattern_width)
                right_idx = min(w - 1, i + pattern_width)
                
                if left_idx < i and not anomalies[left_idx]:
                    cleaned[:, i] = roi[:, left_idx]
                elif right_idx > i and not anomalies[right_idx]:
                    cleaned[:, i] = roi[:, right_idx]
                else:
                    # Use average of surrounding columns
                    surrounding = []
                    for j in range(max(0, i - pattern_width), min(w, i + pattern_width + 1)):
                        if not anomalies[j]:
                            surrounding.append(roi[:, j])
                    if surrounding:
                        cleaned[:, i] = np.mean(surrounding, axis=0)
        
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def fix_blur(self, image):
        """Two-stage deblurring"""
        # First stage: Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Check if more sharpening needed
        laplacian_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if laplacian_var < 100:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            return cv2.filter2D(unsharp, -1, kernel)
        return unsharp

    def fix_contrast(self, image):
        """Contrast enhancement without over-processing"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def fix_rotation(self, image, barcode_region):
        """Rotation correction with proper padding"""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w]
        
        # Add padding before rotation
        pad = int(max(w, h) * 0.2)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, 
                                  cv2.BORDER_CONSTANT, value=255)
        
        edges = cv2.Canny(padded, 50, 150)
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
                center = (padded.shape[1] // 2, padded.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(padded, matrix, padded.shape,
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
                
                # Find actual barcode area in rotated image
                edges = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges > 0))
                if len(coords):
                    x, y, w, h = cv2.boundingRect(coords)
                    return rotated[y:y+h, x:x+w]
        
        return roi

    def binarize(self, image):
        """Careful binarization to avoid holes"""
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 5)
        return thresh

    def preprocess_image(self, image):
        """Improved preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Find barcode region first
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return gray
        
        # Detect problems
        problems = self.detect_image_problems(gray, barcode_region)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)
        
        # Process the barcode region
        if 'foreign_object' in problems:
            gray = self.remove_foreign_object(gray, barcode_region)
            self.add_step("Removed Foreign Object", gray)
        
        if 'blur' in problems:
            gray = self.fix_blur(gray)
            self.add_step("Fixed Blur", gray)
        
        if 'low_contrast' in problems:
            gray = self.fix_contrast(gray)
            self.add_step("Enhanced Contrast", gray)
        
        # Binarize
        binary = self.binarize(gray)
        self.add_step("Binarized", binary)
        
        # Fix rotation last to avoid losing barcode parts
        if 'rotation' in problems:
            binary = self.fix_rotation(binary, barcode_region)
            self.add_step("Fixed Rotation", binary)
        
        return binary

    def visualize_steps(self):
        """Visualization with proper layout"""
        n_steps = len(self.steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (description, img) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.imshow(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     cmap='gray')
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
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()