import os
import cv2
import numpy as np

# Matplotlib in non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tkinter import filedialog
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# -------------------------------------------------------------------
# Set up logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------

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
        return p95 - p5

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
        if image is None:
            return 0.0, {}
        metrics = {
            'edge_score': cls.calculate_edge_quality(image),
            'contrast_score': cls.calculate_contrast_quality(image),
            'pattern_score': cls.calculate_pattern_quality(image),
            'noise_score': cls.calculate_noise_quality(image)
        }
        total_score = (
            metrics['edge_score'] * 0.4 +
            metrics['contrast_score'] * 0.3 +
            metrics['pattern_score'] * 0.2 +
            metrics['noise_score'] * 0.1
        )
        return total_score, metrics

# -------------------------------------------------------------------
# Common Utility Functions
# -------------------------------------------------------------------

def estimate_bar_thickness(image: np.ndarray) -> Tuple[int, int]:
    """Estimate barcode bar thickness."""
    try:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        mid_slice = sobelx.shape[0] // 4
        mid_end = 3 * sobelx.shape[0] // 4
        for row in sobelx[mid_slice:mid_end]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3, 9  # Default fallback

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        return min(min_width, 10), min(pattern_width, 30)
    except Exception as e:
        logger.error(f"Error in estimate_bar_thickness: {str(e)}")
        return 3, 9  # Default fallback

def get_anomalous_regions(anomalies: np.ndarray, bar_thickness: int) -> List[List[int]]:
    """Group contiguous anomalous regions."""
    regions = []
    if len(anomalies) == 0:
        return regions
    try:
        current_region = [anomalies[0]]
        for i in range(1, len(anomalies)):
            if (
                anomalies[i] == anomalies[i-1] + 1 and
                anomalies[i] - current_region[0] < bar_thickness * 2
            ):
                current_region.append(anomalies[i])
            else:
                regions.append(current_region)
                current_region = [anomalies[i]]
        regions.append(current_region)
    except Exception as e:
        logger.error(f"Error in get_anomalous_regions: {str(e)}")
    return regions

def detect_barcode_region(image: np.ndarray, min_region_size: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """Find the actual barcode region in the image."""
    try:
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
            if w < min_region_size or h < min_region_size:
                continue
            aspect_ratio = w / float(h)
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None
        return max(
            valid_regions,
            key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]])
        )
    except Exception as e:
        logger.error(f"Error in detect_barcode_region: {str(e)}")
        return None

# -------------------------------------------------------------------
# Pipeline Base Class
# -------------------------------------------------------------------

class PipelineBase:
    """Base class for all barcode processing pipelines."""
    def __init__(self):
        self.steps = []
        self.problems = []

    def add_step(self, description: str, image: np.ndarray):
        """Add a processing step."""
        try:
            if image is not None:
                self.steps.append((description, image.copy()))
        except Exception as e:
            logger.error(f"Error adding step {description}: {str(e)}")

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process the image. Must be implemented by subclasses."""
        raise NotImplementedError

    def preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        try:
            return (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) > 2 else image.copy()
            )
        except Exception as e:
            logger.error(f"Error in preprocess_grayscale: {str(e)}")
            return image.copy()

    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
        """Detect problems in the image. Should be implemented by subclasses."""
        raise NotImplementedError

# -------------------------------------------------------------------
# Pipeline 1: Pattern-Based Approach
# -------------------------------------------------------------------

class Pipeline1(PipelineBase):
    def __init__(self):
        super().__init__()
        self.min_bar_width = None
        self.pattern_width = None

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

        return min(min_width, 3), min(pattern_width, 9)

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

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def detect_and_log_problems(self, image):
        problems = []
        gray = self.preprocess_grayscale(image)

        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y:y+h, x:x+w]

        # Get pattern info first
        self.min_bar_width, self.pattern_width = self.estimate_bar_thickness(roi)

        # Foreign object detection
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)
        if np.any(np.abs(vertical_profile - v_mean) > 2.5 * v_std):
            problems.append('foreign_object')

        # Rotation detection (do this early)
        edges = cv2.Canny(roi, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        # Blur detection using variance of Laplacian
        laplacian_var = self.variance_of_laplacian(roi)
        if laplacian_var < 80:
            problems.append('blur')

        # Low contrast
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 50:
            problems.append('low_contrast')

        # Enhanced salt and pepper detection
        median = cv2.medianBlur(roi, 3)
        diff = np.abs(roi - median)
        if np.percentile(diff, 95) > 40:
            problems.append('salt_pepper')

        # Frequency domain
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        if np.std(magnitude_spectrum) > 45:
            problems.append('frequency_domain')

        return problems, barcode_region

    def remove_foreign_object(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Get vertical profile
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)

        # Get smallest bar width
        min_bar_width, _ = self.estimate_bar_thickness(roi)
        min_bar_width = min(min_bar_width, 3)

        # Create vertical kernel
        kernel = np.ones((h, min_bar_width)) / h

        # Detect anomalies
        anomalies = np.abs(vertical_profile - v_mean) > 2.5 * v_std

        # Apply vertical averaging only on anomalous regions
        cleaned = roi.copy()
        for i in range(w - min_bar_width + 1):
            if anomalies[i]:
                region = roi[:, i:i+min_bar_width]
                avg_val = np.mean(region, axis=0)
                cleaned[:, i:i+min_bar_width] = np.tile(avg_val, (h, 1))

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def remove_salt_pepper(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Multiple passes of median filtering with increasing kernel sizes
        result = cv2.medianBlur(roi, 3)
        result = cv2.medianBlur(result, 5)

        # Check if we need another pass
        noise_level = np.std(result - roi)
        if noise_level > 30:
            result = cv2.medianBlur(result, 3)

        result_img = image.copy()
        result_img[y:y+h, x:x+w] = result
        return result_img

    def fix_rotation(self, image):
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
                matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)
                return cv2.warpAffine(image, matrix, (cols, rows),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=255)
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

        # Single pass gentle unsharp masking
        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.3, gaussian, -0.3, 0)

        result = image.copy()
        result[y:y+h, x:x+w] = unsharp
        return result

    def fix_frequency_domain(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2

        # Create high-pass filter mask
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        x_grid, y_grid = np.ogrid[:rows, :cols]
        mask_area = (x_grid - crow)**2 + (y_grid - ccol)**2 <= r*r
        mask[mask_area] = 0

        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
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

    def find_optimal_crop(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add padding
        pad_y_top = int(0.05 * h)
        pad_y_bottom = int(0.2 * h)  # More padding for numbers
        pad_x = int(0.05 * w)

        # Ensure we don't exceed image boundaries
        x = max(0, x - pad_x)
        y = max(0, y - pad_y_top)
        w = min(binary.shape[1] - x, w + 2*pad_x)
        h = min(binary.shape[0] - y, h + pad_y_top + pad_y_bottom)

        return binary[y:y+h, x:x+w]

    def final_cleanup(self, binary):
        min_width = 1  # Always use 1-pixel width
        h = binary.shape[0]

        # Apply vertical averaging
        cleaned = np.zeros_like(binary)
        for i in range(binary.shape[1]):
            col = binary[:, i]
            cleaned[:, i] = (np.mean(col) > 127) * 255

        return cleaned

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        gray = self.preprocess_grayscale(image)
        problems, barcode_region = self.detect_and_log_problems(gray)
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

        # Fix frequency domain issues
        if 'frequency_domain' in problems:
            roi = self.fix_frequency_domain(roi, (0, 0, w, h))
            self.add_step("Fixed Frequency Domain", roi)

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

        # Find optimal crop
        cropped = self.find_optimal_crop(binary)
        self.add_step("Cropped", cropped)

        # Final vertical averaging filter
        final = self.final_cleanup(cropped)
        self.add_step("Final Cleanup", final)

        return final
# -------------------------------------------------------------------
# Pipeline 2: Morphological Approach
# -------------------------------------------------------------------

class Pipeline2(PipelineBase):
    """Morphological approach focusing on shape and structure."""
    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int,int,int,int]]]:
        try:
            region = detect_barcode_region(image)
            if region is None:
                return [], None
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            problems = []

            # Check structure using morphological gradient
            kernel = np.ones((3,3), np.uint8)
            gradient = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel)
            if np.mean(gradient) < 20:
                problems.append('structure')

            # Check noise using top-hat transform
            tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
            if np.mean(tophat) > 30:
                problems.append('noise')

            return problems, region
        except Exception as e:
            logger.error(f"Error in Pipeline2 problem detection: {str(e)}")
            return [], None

    def _apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        try:
            kernel = np.ones((3,3), np.uint8)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed
        except Exception as e:
            logger.error(f"Error in morphological operations: {str(e)}")
            return image

    def _enhance_contrast(self, image: np.ndarray)->np.ndarray:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image)
        except Exception as e:
            logger.error(f"Error in contrast enhancement: {str(e)}")
            return image

    def process(self, image: np.ndarray)->np.ndarray:
        try:
            self.steps = []
            self.add_step("Original Image", image)

            gray = self.preprocess_grayscale(image)
            probs, region = self.detect_and_log_problems(gray)
            self.add_step("Detected Problems: "+", ".join(probs), gray)

            if region is None:
                return gray

            x, y, w, h = region
            roi = gray[y:y+h, x:x+w]

            processed = self._apply_morphological_operations(roi)
            self.add_step("Morphological Processing", processed)

            enhanced = self._enhance_contrast(processed)
            self.add_step("Enhanced", enhanced)

            result = gray.copy()
            result[y:y+h, x:x+w] = enhanced
            return result
        except Exception as e:
            logger.error(f"Error in Pipeline2 process: {str(e)}")
            return image

# -------------------------------------------------------------------
# Pipeline 3: Frequency-Domain Approach
# -------------------------------------------------------------------

class Pipeline3(PipelineBase):
    """Frequency domain approach focusing on noise reduction and pattern enhancement."""
    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
        try:
            region = detect_barcode_region(image)
            if region is None:
                return [], None
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            problems = []

            f_transform = np.fft.fft2(roi)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20*np.log(np.abs(f_shift)+1)

            if np.std(magnitude_spectrum) > 45:
                problems.append('frequency_domain')

            # Check for periodic noise
            hf = magnitude_spectrum[0:magnitude_spectrum.shape[0]//4, :]
            if np.max(hf) > 150:
                problems.append('periodic_noise')

            return problems, region
        except Exception as e:
            logger.error(f"Error in Pipeline3 problem detection: {str(e)}")
            return [], None

    def _fix_frequency_domain(self, image: np.ndarray)->np.ndarray:
        try:
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            rows, cols = image.shape
            crow, ccol = rows//2, cols//2
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            x_grid, y_grid = np.ogrid[:rows, :cols]
            mask_area = (x_grid - crow)**2 + (y_grid - ccol)**2 <= r*r
            mask[mask_area] = 0

            f_shift = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in frequency domain fix: {str(e)}")
            return image

    def process(self, image: np.ndarray)->np.ndarray:
        try:
            self.steps = []
            self.add_step("Original Image", image)

            gray = self.preprocess_grayscale(image)
            probs, region = self.detect_and_log_problems(gray)
            self.add_step("Detected Problems: " + ", ".join(probs), gray)

            if region is None:
                return gray

            x, y, w, h = region
            roi = gray[y:y+h, x:x+w]

            processed = roi.copy()
            if 'frequency_domain' in probs or 'periodic_noise' in probs:
                processed = self._fix_frequency_domain(roi)
                self.add_step("Frequency Domain Processing", processed)

            result = gray.copy()
            result[y:y+h, x:x+w] = processed
            return result
        except Exception as e:
            logger.error(f"Error in Pipeline3 process: {str(e)}")
            return image

# -------------------------------------------------------------------
# Multi-Pipeline Manager
# -------------------------------------------------------------------

class BarcodeProcessor:
    """Manager class for running and evaluating multiple pipelines."""
    def __init__(self):
        self.pipeline1 = Pipeline1()
        self.pipeline2 = Pipeline2()
        self.pipeline3 = Pipeline3()
        self.all_results: List[PipelineResult] = []
        self.best_result: Optional[PipelineResult] = None
        self.original_image: Optional[np.ndarray] = None

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image through all pipelines and pick best result."""
        try:
            self.original_image = image.copy()
            self.all_results = []

            # Run each pipeline
            for pipeline, name in [
                (self.pipeline1, "Pipeline 1"),
                (self.pipeline2, "Pipeline 2"),
                (self.pipeline3, "Pipeline 3")
            ]:
                logger.info(f"Running {name}...")
                # Ensure input to Pipeline1 is grayscale
                gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
                result = pipeline.process(gray_image)
                score, metrics = BarcodeImageQuality.evaluate_quality(result)
                logger.info(f"{name} completed with score: {score:.2f}")

                p_result = PipelineResult(
                    name=name,
                    quality_score=score,
                    processed_image=result,
                    steps=pipeline.steps,
                    metrics=metrics
                )
                self.all_results.append(p_result)

            if not self.all_results:
                logger.warning("No pipeline produced results")
                return image

            self.best_result = max(self.all_results, key=lambda x: x.quality_score)
            logger.info(f"Best result: {self.best_result.name} "
                        f"(score: {self.best_result.quality_score:.2f})")
            return self.best_result.processed_image
        except Exception as e:
            logger.error(f"Error in main processing: {str(e)}")
            return image

    def visualize_all_results(self):
        """Visualize pipeline steps & final results."""
        try:
            if not self.all_results or self.original_image is None:
                logger.warning("No results to visualize")
                return

            max_steps = max(len(r.steps) for r in self.all_results)
            rows = 5   # Row0 => Original
                       # Row1 => Pipeline1
                       # Row2 => Pipeline2
                       # Row3 => Pipeline3
                       # Row4 => Winner
            cols = max_steps

            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

            # Row 0 => Original
            axes[0,0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            axes[0,0].set_title("Original Image")
            axes[0,0].axis('off')
            for c in range(1, cols):
                axes[0,c].axis('off')

            # Pipeline results (Rows 1,2,3)
            for row_index, result in enumerate(self.all_results, start=1):
                # Place final pipeline image in first col
                axes[row_index, 0].imshow(result.processed_image, cmap='gray')
                axes[row_index, 0].set_title(result.name)
                axes[row_index, 0].axis('off')

                # Steps
                for step_index, (desc, step_img) in enumerate(result.steps, start=1):
                    if step_index < cols:
                        axes[row_index, step_index].imshow(step_img, cmap='gray')
                        axes[row_index, step_index].set_title(desc)
                        axes[row_index, step_index].axis('off')

            # Final row => Winner
            if self.best_result:
                axes[4,0].imshow(self.best_result.processed_image, cmap='gray')
                axes[4,0].set_title(f"Winner: {self.best_result.name}\n"
                                    f"Score: {self.best_result.quality_score:.2f}")
                axes[4,0].axis('off')

                # Hide extra cells in last row
                for c in range(1, cols):
                    axes[4,c].axis('off')
            else:
                # If no best_result, hide entire last row
                for c in range(cols):
                    axes[4,c].axis('off')

            plt.tight_layout()
            plt.savefig("pipeline_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Results saved to pipeline_comparison.png")
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
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
            print("\nVisualization saved as 'pipeline_comparison.png'")
        else:
            print("Failed to load image.")
    else:
        print("No image selected.")