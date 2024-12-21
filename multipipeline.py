import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks
from typing import List, Tuple, Optional

class BarcodeImageQuality:
    """Class to evaluate barcode image quality"""
    @staticmethod
    def calculate_edge_quality(image: np.ndarray) -> float:
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.mean(edges > 0)
        edge_coherence = cv2.Laplacian(image, cv2.CV_64F).var()
        return edge_density * 100 + edge_coherence * 0.1

    @staticmethod
    def calculate_contrast_quality(image: np.ndarray) -> float:
        p5, p95 = np.percentile(image, [5, 95])
        contrast_range = p95 - p5
        return contrast_range

    @staticmethod
    def calculate_pattern_quality(image: np.ndarray) -> float:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        pattern_strength = np.mean(np.abs(sobelx))
        return pattern_strength

    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        noise = np.abs(image.astype(float) - blurred.astype(float))
        return 255 - np.mean(noise)

    @classmethod
    def evaluate_quality(cls, image: np.ndarray) -> float:
        if image is None:
            return 0.0
        
        edge_score = cls.calculate_edge_quality(image)
        contrast_score = cls.calculate_contrast_quality(image)
        pattern_score = cls.calculate_pattern_quality(image)
        noise_score = cls.calculate_noise_quality(image)
        
        # Weighted combination
        return (edge_score * 0.4 + 
                contrast_score * 0.3 + 
                pattern_score * 0.2 + 
                noise_score * 0.1)

class GeneralPipeline1:
    """Pipeline 1: Based on codebase 3's approach"""
    def __init__(self):
        self.steps = []

    def add_step(self, description: str, image: np.ndarray) -> None:
        self.steps.append((description, image.copy()))

    def detect_barcode_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        edges = cv2.Canny(image, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Initial denoising
        denoised = cv2.medianBlur(gray, 3)
        denoised = cv2.GaussianBlur(denoised, (3,3), 0)
        self.add_step("Denoised", denoised)
        
        # Detect barcode region
        region = self.detect_barcode_region(denoised)
        if region is None:
            return denoised
            
        x, y, w, h = region
        roi = denoised[y:y+h, x:x+w]
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        self.add_step("Enhanced", enhanced)
        
        # Binarization
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.add_step("Binary", binary)
        
        # Final cleanup
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.add_step("Final", cleaned)
        
        return cleaned

class GeneralPipeline2:
    """Pipeline 2: Based on codebase 6's morphological approach"""
    def __init__(self):
        self.steps = []

    def add_step(self, description: str, image: np.ndarray) -> None:
        self.steps.append((description, image.copy()))

    def detect_barcode_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        blurred = cv2.GaussianBlur(gradx, (9,9), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Initial morphological operations
        kernel_noise = np.ones((3,3), np.uint8)
        denoised = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_noise)
        self.add_step("Denoised", denoised)
        
        # Detect barcode region
        region = self.detect_barcode_region(denoised)
        if region is None:
            return denoised
            
        x, y, w, h = region
        roi = denoised[y:y+h, x:x+w]
        
        # Vertical pattern enhancement
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
        enhanced = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_vertical)
        self.add_step("Enhanced", enhanced)
        
        # Binarization
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 51, 5
        )
        self.add_step("Binary", binary)
        
        # Final vertical pattern emphasis
        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_final)
        self.add_step("Final", cleaned)
        
        return cleaned

class GeneralPipeline3:
    """Pipeline 3: Based on codebase 21's quality-based approach"""
    def __init__(self):
        self.steps = []

    def add_step(self, description: str, image: np.ndarray) -> None:
        self.steps.append((description, image.copy()))

    def estimate_bar_width(self, image: np.ndarray) -> int:
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
        if lines is None:
            return 5
            
        widths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:  # Nearly vertical lines
                widths.append(abs(x2 - x1))
                
        return max(3, int(np.median(widths))) if widths else 5

    def detect_barcode_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        
        blur = cv2.GaussianBlur(gradx, (21,21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 10:
                continue
            aspect = w/float(h)
            if 1.5 < aspect < 5:
                valid_regions.append((x,y,w,h))
                
        if not valid_regions:
            return None
            
        return max(valid_regions, key=lambda r: r[2] * r[3])

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Initial cleanup with quality check
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        if BarcodeImageQuality.evaluate_quality(denoised) < BarcodeImageQuality.evaluate_quality(gray):
            denoised = gray
        self.add_step("Denoised", denoised)
        
        # Detect barcode region
        region = self.detect_barcode_region(denoised)
        if region is None:
            return denoised
            
        x, y, w, h = region
        roi = denoised[y:y+h, x:x+w]
        
        # Estimate bar width for kernel size
        bar_width = self.estimate_bar_width(roi)
        
        # Create vertical averaging kernel
        kernel = np.ones((roi.shape[0], bar_width)) / roi.shape[0]
        averaged = cv2.filter2D(roi, -1, kernel)
        self.add_step("Averaged", averaged)
        
        # Binarization with pre-normalization
        p2, p98 = np.percentile(averaged, (2,98))
        normalized = np.interp(averaged, (p2,p98), (0,255)).astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 5
        )
        self.add_step("Binary", binary)
        
        # Final cleanup
        kernel_cleanup = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_cleanup)
        self.add_step("Final", cleaned)
        
        return cleaned

class BarcodeProcessor:
    """Main processor that manages multiple pipelines"""
    def __init__(self):
        self.pipeline1 = GeneralPipeline1()
        self.pipeline2 = GeneralPipeline2()
        self.pipeline3 = GeneralPipeline3()
        self.steps = []

    def add_step(self, description: str, image: np.ndarray) -> None:
        self.steps.append((description, image.copy()))

    def process(self, image: np.ndarray) -> np.ndarray:
        # Process image through each pipeline
        result1 = self.pipeline1.process(image.copy())
        result2 = self.pipeline2.process(image.copy())
        result3 = self.pipeline3.process(image.copy())
        
        # Evaluate results
        quality1 = BarcodeImageQuality.evaluate_quality(result1)
        quality2 = BarcodeImageQuality.evaluate_quality(result2)
        quality3 = BarcodeImageQuality.evaluate_quality(result3)
        
        # Select best result
        best_quality = max(quality1, quality2, quality3)
        if best_quality == quality1:
            best_result = result1
            pipeline_steps = self.pipeline1.steps
            method = "Pipeline 1 (Simple)"
        elif best_quality == quality2:
            best_result = result2
            pipeline_steps = self.pipeline2.steps
            method = "Pipeline 2 (Morphological)"
        else:
            best_result = result3
            pipeline_steps = self.pipeline3.steps
            method = "Pipeline 3 (Quality-based)"
            
        # Save steps
        self.steps = []
        self.add_step("Original Image", image)
        for desc, img in pipeline_steps[1:]:  # Skip original image
            self.add_step(f"{method} - {desc}", img)
        
        return best_result

    def visualize_steps(self):
        """Visualize all processing steps"""
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
        for i in range(len(self.steps), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

def main():
    processor = BarcodeProcessor()
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = processor.process(image)
            processor.visualize_steps()

if __name__ == "__main__":
    main()