import os
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from tkinter import filedialog
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple

# -------------------------
# Data Structures
# -------------------------

@dataclass
class PipelineResult:
    """Class to store pipeline results and metrics."""
    name: str
    quality_score: float
    processed_image: np.ndarray
    steps: List[Tuple[str, np.ndarray]]
    metrics: dict

class BarcodeImageQuality:
    """Class to evaluate barcode image quality."""
    @staticmethod
    def calculate_edge_quality(image: np.ndarray) -> float:
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.mean(edges > 0)
        edge_coherence = cv2.Laplacian(image, cv2.CV_64F).var()
        return edge_density * 100 + edge_coherence * 0.1

    @staticmethod
    def calculate_contrast_quality(image: np.ndarray) -> float:
        p5, p95 = np.percentile(image, [5, 95])
        return p95 - p5  # Contrast range

    @staticmethod
    def calculate_pattern_quality(image: np.ndarray) -> float:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        return np.mean(np.abs(sobelx))

    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        noise = np.abs(image.astype(float) - blurred.astype(float))
        return 255 - np.mean(noise)

    @classmethod
    def evaluate_quality(cls, image: np.ndarray) -> Tuple[float, dict]:
        """Compute a weighted total quality score plus individual metrics."""
        if image is None:
            return 0.0, {}

        metrics = {
            'edge_score': cls.calculate_edge_quality(image),
            'contrast_score': cls.calculate_contrast_quality(image),
            'pattern_score': cls.calculate_pattern_quality(image),
            'noise_score': cls.calculate_noise_quality(image)
        }

        # Weighted total
        total_score = (metrics['edge_score'] * 0.4 +
                       metrics['contrast_score'] * 0.3 +
                       metrics['pattern_score'] * 0.2 +
                       metrics['noise_score'] * 0.1)

        return total_score, metrics


# -------------------------
# Pipeline 1 (Simple)
# -------------------------

class GeneralPipeline1:
    """Pipeline 1: Based on codebase 3's approach."""
    def __init__(self):
        self.steps = []

    def add_step(self, description: str, image: np.ndarray):
        self.steps.append((description, image.copy()))

    def detect_barcode_region(self, image: np.ndarray):
        """Enhanced barcode region detection using Sobel & adaptive thresholding."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        thresh = cv2.adaptiveThreshold(gradx, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue

            aspect_ratio = w / float(h)
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()

        denoised = cv2.medianBlur(gray, 3)
        self.add_step("Denoised", denoised)

        region = self.detect_barcode_region(denoised)
        if region is None:
            print("Barcode region not detected in Pipeline 1.")
            return denoised
        x, y, w, h = region

        roi = denoised[y:y+h, x:x+w]
        self.add_step("ROI", roi)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)
        self.add_step("Enhanced Contrast", enhanced)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.add_step("Binarized", binary)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.add_step("Cleaned", cleaned)

        return cleaned


# -------------------------
# Pipeline 2 (Morphological)
# -------------------------

class GeneralPipeline2:
    """Pipeline 2: Based on codebase 6's morphological approach."""
    def __init__(self):
        self.steps = []

    def add_step(self, description: str, image: np.ndarray):
        self.steps.append((description, image.copy()))

    def detect_barcode_region(self, image: np.ndarray):
        """Enhanced barcode region detection."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        blur = cv2.GaussianBlur(gradx, (21,21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50) / (w*h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image.copy()
        region = self.detect_barcode_region(gray)
        if region is None:
            print("Barcode region not detected in Pipeline 2.")
            return gray
        x, y, w, h = region

        roi = gray[y:y+h, x:x+w]
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        self.add_step("Morphological Opening", opening)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(opening)
        self.add_step("CLAHE Enhanced", enhanced)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.add_step("Binarized", binary)

        return binary


# -------------------------
# Pipeline 3 (Quality-based)
# -------------------------

class GeneralPipeline3:
    """Pipeline 3: Based on codebase 21's approach, now with detect_barcode_region."""
    def __init__(self):
        self.steps = []

    def add_step(self, description: str, image: np.ndarray):
        self.steps.append((description, image.copy()))

    # ---------------------------------------------------
    # Step 1: Detect barcode region
    # ---------------------------------------------------
    def detect_barcode_region(self, image: np.ndarray):
        """Detect barcode region in a similar way to pipeline2 (Sobel+Gaussian+Otsu)."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50)/(w*h)
            if 1.5 < aspect_ratio < 5 and edge_density>0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    # ---------------------------------------------------
    # Step 2: Problem detection
    # ---------------------------------------------------
    def detect_image_problems(self, image: np.ndarray):
        """Detect image problems, returning a list plus the region."""
        problems = []
        gray = image.copy()

        region = self.detect_barcode_region(gray)
        if region is None:
            return problems, (0,0,gray.shape[1],gray.shape[0])
        x, y, w, h = region

        roi = gray[y:y+h, x:x+w]

        # Example: check “low_contrast”
        p5, p95 = np.percentile(roi, [5,95])
        if (p95 - p5) < 80:
            problems.append('low_contrast')

        # Just a placeholder: you can re-add salt/pepper, blur, rotation, etc. here
        # or copy your existing logic from codebase21 detect_image_problems.

        return problems, (x, y, w, h)

    # ---------------------------------------------------
    # Step 3: Fixes
    # ---------------------------------------------------
    def remove_foreign_object(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        # Placeholder example
        return image

    def fix_frequency_domain(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        # Placeholder example
        return image

    def fix_compression_artifacts(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        # Placeholder example
        return image

    def remove_salt_and_pepper(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        # Placeholder example
        return image

    def fix_blur(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        # Placeholder example
        return image

    def fix_contrast(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        # Simple CLAHE approach
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def binarize(self, image: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0,255)).astype(np.uint8)
        binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 51, 5)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def final_barcode_cleanup(self, binary: np.ndarray) -> np.ndarray:
        # Just a quick example
        return binary

    # ---------------------------------------------------
    # Step 4: Main pipeline
    # ---------------------------------------------------
    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        problems, region = self.detect_image_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        x, y, w, h = region
        # example fix steps
        if 'low_contrast' in problems:
            fixed_contrast = self.fix_contrast(gray, region)
            self.add_step("Fixed Contrast", fixed_contrast)
            gray = fixed_contrast

        # Binarize
        roi = gray[y:y+h, x:x+w]
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        cleaned = self.final_barcode_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        return cleaned


# -------------------------
# Multi-Pipeline Manager
# -------------------------

@dataclass
class BarcodeProcessor:
    pipeline1: GeneralPipeline1
    pipeline2: GeneralPipeline2
    pipeline3: GeneralPipeline3
    all_results: List[PipelineResult]
    best_result: PipelineResult
    original_image: np.ndarray

    def __init__(self):
        self.pipeline1 = GeneralPipeline1()
        self.pipeline2 = GeneralPipeline2()
        self.pipeline3 = GeneralPipeline3()
        self.all_results = []
        self.best_result = None
        self.original_image = None

    def process(self, image: np.ndarray) -> np.ndarray:
        self.original_image = image.copy()
        self.all_results = []

        for pipeline, name in [
            (self.pipeline1, "Pipeline 1"),
            (self.pipeline2, "Pipeline 2"),
            (self.pipeline3, "Pipeline 3")
        ]:
            result = pipeline.process(image.copy())
            score, metrics = BarcodeImageQuality.evaluate_quality(result)
            p_result = PipelineResult(
                name=name,
                quality_score=score,
                processed_image=result,
                steps=pipeline.steps,
                metrics=metrics
            )
            self.all_results.append(p_result)

        # Pick best
        self.best_result = max(self.all_results, key=lambda x: x.quality_score)
        return self.best_result.processed_image

    def visualize_all_results(self):
        """Create a 5-row figure:
           row=0 => original image
           row=1 => pipeline1 steps
           row=2 => pipeline2 steps
           row=3 => pipeline3 steps
           row=4 => winner pipeline’s final result + metrics."""
        if not self.all_results or self.original_image is None:
            print("No results to visualize")
            return

        # Row 0: original
        # Row 1..3: pipeline steps
        # Row 4: winner

        max_steps = max(len(r.steps) for r in self.all_results)
        rows = 5
        cols = max_steps

        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

        # row=0 => Original image
        axes[0,0].imshow(self.original_image, cmap='gray')
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        for c in range(1, cols):
            axes[0,c].axis('off')

        # row=1..3 => pipeline steps
        for row_index, pipeline_result in enumerate(self.all_results, start=1):
            for col_index in range(cols):
                ax = axes[row_index, col_index]
                if col_index < len(pipeline_result.steps):
                    desc, img = pipeline_result.steps[col_index]
                    if len(img.shape) == 2:
                        ax.imshow(img, cmap='gray')
                    else:
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"{pipeline_result.name}: {desc}", fontsize=8)
                ax.axis('off')

        # row=4 => winner in col=0, metrics in col=1
        winner_img = self.best_result.processed_image
        axes[4,0].imshow(winner_img, cmap='gray')
        axes[4,0].set_title(f"Winner: {self.best_result.name}\n"
                            f"Score: {self.best_result.quality_score:.2f}", fontsize=8)
        axes[4,0].axis('off')

        # Show winner's metrics in col=1
        metrics_text = "\n".join([
            f"Edge Score: {self.best_result.metrics['edge_score']:.2f}",
            f"Contrast Score: {self.best_result.metrics['contrast_score']:.2f}",
            f"Pattern Score: {self.best_result.metrics['pattern_score']:.2f}",
            f"Noise Score: {self.best_result.metrics['noise_score']:.2f}"
        ])
        axes[4,1].text(0.05, 0.5, metrics_text, fontsize=10, va='center')
        axes[4,1].axis('off')

        # Hide any remaining columns in row=4
        for c in range(2, cols):
            axes[4,c].axis('off')

        plt.tight_layout()
        plt.savefig("pipeline_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def print_statistics(self):
        """Print detailed stats for all pipelines."""
        print("\n=== Pipeline Processing Statistics ===")
        for result in self.all_results:
            print(f"\n{result.name}:")
            print(f"Overall Quality Score: {result.quality_score:.2f}")
            print("Detailed Metrics:")
            for metric_name, metric_value in result.metrics.items():
                print(f"  {metric_name}: {metric_value:.2f}")

        print(f"\nWinning Pipeline: {self.best_result.name}")
        print(f"Winning Score: {self.best_result.quality_score:.2f}")


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    processor = BarcodeProcessor()
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = processor.process(image)
            processor.visualize_all_results()
            processor.print_statistics()
            print("\nVisualization saved as 'pipeline_comparison.png'")
        else:
            print("Failed to load image.")
    else:
        print("No image selected.")
