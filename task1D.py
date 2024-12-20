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

    def initial_cleanup(self, image):
        """Basic image enhancement - Applied to all images"""
        # Initial mild noise reduction
        denoised = cv2.medianBlur(image, 3)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        
        # Basic OTSU thresholding for normalization
        _, normalized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return normalized

    def check_rotation_angle(self, image):
        """Precisely measure rotation angle"""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(image.shape)//3)
        
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
                # Only return if significant rotation
                if abs(median_angle) > 5:
                    return median_angle
        
        return 0

    def fix_rotation(self, image, angle):
        """Fix rotation with proper padding"""
        if abs(angle) <= 5:  # Don't rotate if angle is small
            return image
            
        # Add padding before rotation
        pad = int(max(image.shape) * 0.3)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=255)
        
        # Get rotation matrix
        rows, cols = padded.shape
        center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((rows * sin) + (cols * cos))
        new_h = int((rows * cos) + (cols * sin))
        
        # Adjust transformation matrix
        M[0, 2] += (new_w/2) - center[0]
        M[1, 2] += (new_h/2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(padded, M, (new_w, new_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=255)
        
        # Find content bbox
        coords = np.column_stack(np.where(rotated < 250))
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        # Crop to content
        cropped = rotated[y_min:y_max+1, x_min:x_max+1]
        return cropped

    def detect_barcode_region(self, image):
        """Find the actual barcode region"""
        # Enhance vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        # Find regions with high concentration of vertical lines
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Filter contours by aspect ratio and area
        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 10:  # Too small
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))
                
        if not valid_regions:
            return None
            
        # Return region with highest edge density
        x, y, w, h = max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))
        
        # Add padding
        pad_x = int(0.1 * w)
        pad_y = int(0.1 * h)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(image.shape[1] - x, w + 2*pad_x)
        h = min(image.shape[0] - y, h + 2*pad_y)
        
        return x, y, w, h

    def remove_foreign_object(self, image):
        """Remove foreign objects using vertical pattern analysis"""
        # Estimate bar width
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                               minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            widths = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10:  # Nearly vertical lines
                    widths.append(abs(x2 - x1))
            bar_width = max(3, int(np.median(widths)) if widths else 5)
        else:
            bar_width = 5
            
        # Create vertical averaging kernels
        kernel1 = np.ones((image.shape[0], bar_width)) / image.shape[0]
        kernel2 = np.ones((image.shape[0], bar_width * 3)) / image.shape[0]
        kernel3 = np.ones((image.shape[0], 1000)) / image.shape[0]  # Very tall kernel
        
        # Apply kernels progressively
        filtered1 = cv2.filter2D(image, -1, kernel1)
        filtered2 = cv2.filter2D(filtered1, -1, kernel2)
        filtered3 = cv2.filter2D(filtered2, -1, kernel3)
        
        # Binarize result
        _, result = cv2.threshold(filtered3, 127, 255, cv2.THRESH_BINARY)
        
        return result

    def fix_frequency_domain(self, image):
        """Fix frequency domain issues"""
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

    def fix_contrast(self, image):
        """Fix contrast based on image statistics"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        # Only enhance if needed
        if std_val < 30:  # Low contrast
            if mean_val < 128:  # Dark image
                # Stretch histogram for dark images
                p2, p98 = np.percentile(image, (2, 98))
                enhanced = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
            else:  # Bright image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
            return enhanced
        return image

    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        self.steps = []
        self.add_step("Original Image", image)
        
        # Stage 1: General Enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        enhanced = self.initial_cleanup(gray)
        self.add_step("Initial Enhancement", enhanced)
        
        # Stage 2: Core Problems
        # Check rotation first
        angle = self.check_rotation_angle(enhanced)
        if abs(angle) > 5:
            enhanced = self.fix_rotation(enhanced, angle)
            self.add_step("Fixed Rotation", enhanced)
        
        # Find barcode region
        region = self.detect_barcode_region(enhanced)
        if region is not None:
            x, y, w, h = region
            enhanced = enhanced[y:y+h, x:x+w]
            self.add_step("Cropped Barcode", enhanced)
        
        # Stage 3: Barcode-Specific Processing
        # Remove foreign objects
        enhanced = self.remove_foreign_object(enhanced)
        self.add_step("Removed Foreign Objects", enhanced)
        
        # Fix frequency domain issues
        enhanced = self.fix_frequency_domain(enhanced)
        self.add_step("Fixed Frequency Domain", enhanced)
        
        # Final contrast enhancement
        enhanced = self.fix_contrast(enhanced)
        self.add_step("Enhanced Contrast", enhanced)
        
        # Stage 4: Final Cleanup
        # Final binarization
        _, binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        self.add_step("Final Result", binary)
        
        return binary

    def visualize_steps(self):
        """Visualize processing steps"""
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