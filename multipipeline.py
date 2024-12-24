import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    """Enhanced quality metrics for barcode images."""
    @staticmethod
    def calculate_edge_quality(image: np.ndarray) -> float:
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.mean(edges > 0)
        edge_coherence = cv2.Laplacian(image, cv2.CV_64F).var()
        vertical_gradient = np.mean(np.abs(np.diff(image, axis=1)))
        # Bonus for strong vertical patterns
        if vertical_gradient > 50:
            return (edge_density * 100 + edge_coherence * 0.1) * 1.2
        return edge_density * 100 + edge_coherence * 0.1

    @staticmethod
    def calculate_contrast_quality(image: np.ndarray) -> float:
        p5, p95 = np.percentile(image, [5, 95])
        mean_intensity = np.mean(image)
        contrast = p95 - p5
        # Don't heavily penalize dark/bright images
        if mean_intensity < 30 or mean_intensity > 225:
            return contrast * 0.8
        return contrast

    @staticmethod
    def calculate_pattern_quality(image: np.ndarray) -> float:
        # Enhanced pattern detection focusing on vertical lines
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        pattern_score = np.mean(np.abs(sobelx))
        # Add bonus for clear barcode patterns
        if pattern_score > 50:
            pattern_score *= 1.2
        return pattern_score

    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
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
            'noise_score': cls.calculate_noise_quality(image),
        }

        # Enhanced weighting
        total_score = (
            metrics['edge_score'] * 0.4
            + metrics['contrast_score'] * 0.3
            + metrics['pattern_score'] * 0.2
            + metrics['noise_score'] * 0.1
        )

        # Additional bonus for ideal barcode characteristics
        mean_intensity = np.mean(image)
        if 80 < mean_intensity < 175:  # Good contrast range
            total_score *= 1.1

        return total_score, metrics

# -------------------------------------------------------------------
# Common Utility Functions
# -------------------------------------------------------------------
def estimate_bar_thickness(image: np.ndarray) -> Tuple[int, int]:
    """Estimate the barcode bar thickness."""
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
            return 3, 9

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        return min(min_width, 10), min(pattern_width, 30)

    except Exception as e:
        logger.error(f"Error in estimate_bar_thickness: {str(e)}")
        return 3, 9

def get_anomalous_regions(anomalies: np.ndarray, bar_thickness: int) -> List[List[int]]:
    """Group contiguous anomalous column indices into regions."""
    regions = []
    if len(anomalies) == 0:
        return regions
    try:
        current_region = [anomalies[0]]
        for i in range(1, len(anomalies)):
            if (
                anomalies[i] == anomalies[i - 1] + 1
                and anomalies[i] - current_region[0] < bar_thickness * 2
            ):
                current_region.append(anomalies[i])
            else:
                regions.append(current_region)
                current_region = [anomalies[i]]
        regions.append(current_region)
    except Exception as e:
        logger.error(f"Error in get_anomalous_regions: {str(e)}")
    return regions

def detect_barcode_region(
    image: np.ndarray, min_region_size: int = 10
) -> Optional[Tuple[int, int, int, int]]:
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
            roi = gradx[y : y + h, x : x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        return max(
            valid_regions,
            key=lambda r: np.sum(gradx[r[1] : r[1] + r[3], r[0] : r[0] + r[2]]),
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
        """Add a processing step with a description and a copy of the image."""
        try:
            if image is not None:
                self.steps.append((description, image.copy()))
        except Exception as e:
            logger.error(f"Error adding step {description}: {str(e)}")

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process the image. Must be overridden by subclasses."""
        raise NotImplementedError

    def preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert the image to grayscale if needed."""
        try:
            if len(image.shape) > 2:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.copy()
        except Exception as e:
            logger.error(f"Error in preprocess_grayscale: {str(e)}")
            return image.copy()

    def detect_and_log_problems(
        self, image: np.ndarray
    ) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
        """Detect problems in the image. Must be overridden by subclasses."""
        raise NotImplementedError

# -------------------------------------------------------------------
# Pipeline 1: Pattern-Based Approach
# -------------------------------------------------------------------
class Pipeline1(PipelineBase):
    """Pipeline 1: Pattern-based approach with rotation, foreign object fix, etc."""
    def __init__(self):
        super().__init__()
        self.min_bar_width = None
        self.pattern_width = None
        self.cropped_flag = False

    def estimate_bar_thickness(self, image):
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        for row in sobelx[sobelx.shape[0] // 4 : 3 * sobelx.shape[0] // 4]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 1, 3

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        return min(min_width, 3), min(pattern_width, 9)

    def detect_barcode_region(self, image):
        """Detect barcode region specifically for Pipeline 1."""
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
            roi = gradx[y : y + h, x : x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        largest = max(valid_regions, key=lambda r: r[2] * r[3])
        return largest

    def variance_of_laplacian(self, image):
        """Compute Laplacian variance for blur detection."""
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def detect_and_log_problems(self, image):
        """Detect typical barcode problems: rotation, blur, low contrast, salt/pepper, freq domain."""
        problems = []
        gray = self.preprocess_grayscale(image)

        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return problems, (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y : y + h, x : x + w]

        # Estimate bar thickness
        self.min_bar_width, self.pattern_width = self.estimate_bar_thickness(roi)

        # Foreign object
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)
        if np.any(np.abs(vertical_profile - v_mean) > 2.5 * v_std):
            problems.append('foreign_object')

        # Rotation
        edges = cv2.Canny(roi, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        # Blur detection
        lap_var = self.variance_of_laplacian(roi)
        if lap_var < 80:
            problems.append('blur')

        # Low contrast
        p5, p95 = np.percentile(roi, [5, 95])
        if (p95 - p5) < 50:
            problems.append('low_contrast')

        # Salt & Pepper
        median = cv2.medianBlur(roi, 3)
        diff = np.abs(roi - median)
        if np.percentile(diff, 95) > 30:
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
        roi = image[y : y + h, x : x + w].copy()

        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_mean = np.mean(vertical_profile)

        min_bar_width, _ = self.estimate_bar_thickness(roi)
        min_bar_width = min(min_bar_width, 3)

        kernel = np.ones((h, min_bar_width)) / h
        anomalies = np.abs(vertical_profile - v_mean) > 2.5 * v_std

        cleaned = roi.copy()
        for i in range(w - min_bar_width + 1):
            if anomalies[i]:
                region = roi[:, i : i + min_bar_width]
                avg_val = np.mean(region, axis=0)
                cleaned[:, i : i + min_bar_width] = np.tile(avg_val, (h, 1))

        result = image.copy()
        result[y : y + h, x : x + w] = cleaned
        return result

    def remove_salt_pepper(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        result = cv2.medianBlur(roi, 3)
        result = cv2.medianBlur(result, 5)

        noise_level = np.std(result - roi)
        if noise_level > 20:
            result = cv2.medianBlur(result, 3)

        result_img = image.copy()
        result_img[y : y + h, x : x + w] = result
        return result_img

    def fix_rotation(self, image):
        """Fix rotation using edge detection and angle calculation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use probabilistic Hough transform for better line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) > 5:  # Consider only significant angles
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated

        return image

    def fix_contrast(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)

        result = image.copy()
        result[y : y + h, x : x + w] = enhanced
        return result

    def fix_blur(self, image, barcode_region):
        """Apply gentler unsharp masking to avoid destroying thin bars."""
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        gaussian = cv2.GaussianBlur(roi, (0, 0), 1.5)
        unsharp = cv2.addWeighted(roi, 1.5, gaussian, -0.5, 0)

        result = image.copy()
        result[y : y + h, x : x + w] = unsharp
        return result

    def fix_frequency_domain(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y : y + h, x : x + w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = roi.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), np.uint8)
        r = 30
        x_grid, y_grid = np.ogrid[:rows, :cols]
        mask_area = (x_grid - crow) ** 2 + (y_grid - ccol) ** 2 <= r * r
        mask[mask_area] = 0

        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y : y + h, x : x + w] = cleaned
        return result

    def binarize(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)

        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # Larger window
            5
        )

        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _projection_based_crop(self, binary_image, original_roi):
        """Crop the barcode region based on projections."""
        vertical_projection = np.sum(binary_image == 0, axis=0)
        horizontal_projection = np.sum(binary_image == 0, axis=1)

        vertical_threshold = 0.6 * np.max(vertical_projection)
        horizontal_threshold = 0.6 * np.max(horizontal_projection)

        vertical_indices = np.where(vertical_projection > vertical_threshold)[0]
        horizontal_indices = np.where(horizontal_projection > horizontal_threshold)[0]

        if len(vertical_indices) > 0 and len(horizontal_indices) > 0:
            x_start, x_end = vertical_indices[0], vertical_indices[-1]
            y_start, y_end = horizontal_indices[0], horizontal_indices[-1]
            padding_x = int(0.05 * (x_end - x_start))
            padding_y = int(0.05 * (y_end - y_start))
            x_start = max(0, x_start - padding_x)
            x_end = min(original_roi.shape[1], x_end + padding_x)
            y_start = max(0, y_start - padding_y)
            y_end = min(original_roi.shape[0], y_end + padding_y)

            cropped = original_roi[y_start:y_end, x_start:x_end]
            return cropped
        return original_roi

    def find_optimal_crop(self, binary):
        """Optionally crop the barcode with minimal padding."""
        if self.cropped_flag:
            return binary

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        pad_y_top = int(0.02 * h)
        pad_y_bottom = int(0.02 * h)
        pad_x = int(0.02 * w)

        x = max(0, x - pad_x)
        y = max(0, y - pad_y_top)
        w = min(binary.shape[1] - x, w + 2 * pad_x)
        h = min(binary.shape[0] - y, h + pad_y_top + pad_y_bottom)

        self.cropped_flag = True
        return binary[y : y + h, x : x + w]

    def final_cleanup(self, image):
        """Apply a final cleanup step, considering barcode structure."""
        # Calculate the average intensity of the image
        avg_intensity = np.mean(image)
        
        # Apply morphological closing to enhance barcode structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # If the image is too bright or too dark after enhancement, adjust accordingly
        if avg_intensity > 150:
            # For very bright images, apply a threshold to highlight darker areas (bars)
            _, thresh = cv2.threshold(closed, avg_intensity * 0.7, 255, cv2.THRESH_BINARY)
            return thresh
        elif avg_intensity < 100:
            # For very dark images, enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(closed)
            return enhanced
        else:
            # For moderately intense images, return the enhanced image
            return closed
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the pipeline on an image."""
        self.steps = []
        self.add_step("Original Image", image)

        gray = self.preprocess_grayscale(image)
        problems, barcode_region = self.detect_and_log_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if barcode_region is None:
            return gray

        processed = gray.copy()
        x, y, w, h = barcode_region

        if 'rotation' in problems:
            processed = self.fix_rotation(processed)
            self.add_step("Fixed Rotation", processed)
            new_region = self.detect_barcode_region(processed)
            if new_region:
                barcode_region = new_region
                x, y, w, h = barcode_region

        roi = processed[y : y + h, x : x + w].copy()

        if 'foreign_object' in problems:
            roi = self.remove_foreign_object(roi, (0, 0, w, h))
            self.add_step("Removed Foreign Object", roi)

        if 'salt_pepper' in problems:
            roi = self.remove_salt_pepper(roi, (0, 0, w, h))
            self.add_step("Removed Salt & Pepper", roi)
            # Apply projection-based cropping after salt & pepper removal
            binary_roi = self.binarize(roi)
            cropped_roi = self._projection_based_crop(binary_roi, roi)
            self.add_step("Cropped (Projection)", cropped_roi)
            roi = cropped_roi

        if 'frequency_domain' in problems:
            roi = self.fix_frequency_domain(roi, (0, 0, w, h))
            self.add_step("Fixed Frequency Domain", roi)

        if 'blur' in problems:
            roi = self.fix_blur(roi, (0, 0, w, h))
            self.add_step("Fixed Blur", roi)

        if 'low_contrast' in problems:
            roi = self.fix_contrast(roi, (0, 0, w, h))
            self.add_step("Enhanced Contrast", roi)

        binary = self.binarize(roi)
        self.add_step("Binarized", binary)

        cropped = self.find_optimal_crop(binary)
        self.add_step("Cropped", cropped)

        final = self.final_cleanup(cropped)
        self.add_step("Final Cleanup", final)

        return final

# -------------------------------------------------------------------
# Pipeline 2: Morphological Approach + Pipeline 1 Improvements
# -------------------------------------------------------------------
class Pipeline2(PipelineBase):
    """Morphological approach focusing on shape and structure."""
    def __init__(self):
        super().__init__()
        self.applied_pipeline1_fixes = False  # Flag to track if Pipeline 1 fixes have been applied

    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
        try:
            region = detect_barcode_region(image)
            if region is None:
                return [], None

            x, y, w, h = region
            roi = image[y : y + h, x : x + w]
            problems = []

            kernel = np.ones((3, 3), np.uint8)
            gradient = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel)
            if np.mean(gradient) < 20:
                problems.append('structure')

            tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
            if np.mean(tophat) > 30:
                problems.append('noise')

            return problems, region

        except Exception as e:
            logger.error(f"Error in Pipeline2 problem detection: {str(e)}")
            return [], None

    def _apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Open then close to remove noise and fill small gaps."""
        try:
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed
        except Exception as e:
            logger.error(f"Error in morphological operations: {str(e)}")
            return image

    def fix_rotation(self, image):
        """Fix rotation using edge detection and angle calculation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use probabilistic Hough transform for better line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) > 5:  # Consider only significant angles
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated

        return image

    def fix_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply mild unsharp mask for slight blur correction."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 0.5)  # Reduced sigma for milder blur
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)  # Adjusted weights
        return unsharp

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        except Exception as e:
            logger.error(f"Error in contrast enhancement: {str(e)}")
            return image

    def _projection_based_crop(self, binary_image, original_roi):
        """Crop the barcode region based on projections."""
        vertical_projection = np.sum(binary_image == 0, axis=0)
        horizontal_projection = np.sum(binary_image == 0, axis=1)

        vertical_threshold = 0.6 * np.max(vertical_projection)
        horizontal_threshold = 0.6 * np.max(horizontal_projection)

        vertical_indices = np.where(vertical_projection > vertical_threshold)[0]
        horizontal_indices = np.where(horizontal_projection > horizontal_threshold)[0]

        if len(vertical_indices) > 0 and len(horizontal_indices) > 0:
            x_start, x_end = vertical_indices[0], vertical_indices[-1]
            y_start, y_end = horizontal_indices[0], horizontal_indices[-1]
            padding_x = int(0.05 * (x_end - x_start))
            padding_y = int(0.05 * (y_end - y_start))
            x_start = max(0, x_start - padding_x)
            x_end = min(original_roi.shape[1], x_end + padding_x)
            y_start = max(0, y_start - padding_y)
            y_end = min(original_roi.shape[0], y_end + padding_y)

            cropped = original_roi[y_start:y_end, x_start:x_end]
            return cropped
        return original_roi

    def check_inversion(self, image: np.ndarray) -> np.ndarray:
        """Check for and correct inversion."""
        mean_brightness = np.mean(image)
        if mean_brightness > 127:
            return 255 - image
        return image

    def apply_pipeline1_salt_pepper_fix(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply salt and pepper removal and projection-based cropping from Pipeline 1."""
        x, y, w, h = region
        roi = image[y : y + h, x : x + w].copy()

        # Salt & Pepper Removal (from Pipeline 1)
        roi = cv2.medianBlur(roi, 3)
        roi = cv2.medianBlur(roi, 5)
        noise_level = np.std(roi - image[y:y+h, x:x+w])
        if noise_level > 20:
            roi = cv2.medianBlur(roi, 3)

        # Binarize for cropping
        p2, p98 = np.percentile(roi, (2, 98))
        normalized = np.interp(roi, (p2, p98), (0, 255)).astype(np.uint8)
        binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

        # Projection-based Cropping (from Pipeline 1)
        cropped_roi = self._projection_based_crop(binary, roi)

        return cropped_roi

    def process(self, image: np.ndarray) -> np.ndarray:
        try:
            self.steps = []
            self.add_step("Original Image", image)

            gray = self.preprocess_grayscale(image)
            probs, region = self.detect_and_log_problems(gray)
            self.add_step("Detected Problems: " + ", ".join(probs), gray)

            if region is None:
                return gray

            x, y, w, h = region
            roi = gray[y : y + h, x : x + w]

            # Apply Pipeline 1's salt & pepper fix if detected
            if 'salt_pepper' in probs and not self.applied_pipeline1_fixes:
                roi = self.apply_pipeline1_salt_pepper_fix(roi, (0, 0, w, h))
                self.add_step("Applied Pipeline 1 Salt & Pepper Fix", roi)
                self.applied_pipeline1_fixes = True

            # Apply Pipeline 1's rotation fix if detected and not already applied
            if 'rotation' in probs and not self.applied_pipeline1_fixes:
                roi = self.fix_rotation(roi)
                self.add_step("Applied Pipeline 1 Rotation Fix", roi)
                new_region = detect_barcode_region(roi)
                if new_region:
                    region = new_region
                    x, y, w, h = region
                    roi = gray[y: y + h, x : x + w]
                self.applied_pipeline1_fixes = True

            processed = self._apply_morphological_operations(roi)
            self.add_step("Morphological Processing", processed)

            if 'blur' in probs:
                processed = self.fix_blur(processed)
                self.add_step("Sharpened", processed)

            enhanced = self._enhance_contrast(processed)
            self.add_step("Enhanced Contrast", enhanced)

            # Projection-based cropping
            binary_roi = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cropped = self._projection_based_crop(binary_roi, enhanced)
            self.add_step("Cropped (Projection)", cropped)

            result = gray.copy()
            result[y : y + h, x : x + w] = cropped

            # Check for inversion and correct if needed
            result = self.check_inversion(result)
            self.add_step("Inversion Check", result)

            return result

        except Exception as e:
            logger.error(f"Error in Pipeline2 process: {str(e)}")
            return image

# -------------------------------------------------------------------
# Pipeline 3: Frequency-Domain Approach + Inversion Fix
# -------------------------------------------------------------------
class Pipeline3(PipelineBase):
    """Frequency domain approach focusing on noise reduction and pattern enhancement."""
    def detect_and_log_problems(
        self, image: np.ndarray
    ) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
        try:
            region = detect_barcode_region(image)
            if region is None:
                return [], None

            x, y, w, h = region
            roi = image[y : y + h, x : x + w]
            problems = []

            f_transform = np.fft.fft2(roi)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

            if np.std(magnitude_spectrum) > 45:
                problems.append('frequency_domain')

            hf = magnitude_spectrum[: magnitude_spectrum.shape[0] // 4, :]
            if np.max(hf) > 150:
                problems.append('periodic_noise')

            return problems, region

        except Exception as e:
            logger.error(f"Error in Pipeline3 problem detection: {str(e)}")
            return [], None

    def _fix_frequency_domain(self, image: np.ndarray) -> np.ndarray:
        try:
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            # Zero-out center frequencies
            center_freq_row = slice(crow - r, crow + r)
            center_freq_col = slice(ccol - r, ccol + r)
            mask[center_freq_row, center_freq_col] = 0

            f_shift_filtered = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        except Exception as e:
            logger.error(f"Error in frequency domain fix: {str(e)}")
            return image

    def check_inversion(self, image: np.ndarray) -> np.ndarray:
        """If the final is mostly bright, invert. Otherwise, keep it."""
        mean_brightness = np.mean(image)
        if mean_brightness > 127:
            return 255 - image
        return image

    def process(self, image: np.ndarray) -> np.ndarray:
        try:
            self.steps = []
            self.add_step("Original Image", image)

            gray = self.preprocess_grayscale(image)
            probs, region = self.detect_and_log_problems(gray)
            self.add_step("Detected Problems: " + ", ".join(probs), gray)

            if region is None:
                return gray

            x, y, w, h = region
            roi = gray[y : y + h, x : x + w]

            processed = roi.copy()
            if 'frequency_domain' in probs or 'periodic_noise' in probs:
                processed = self._fix_frequency_domain(roi)
                self.add_step("Frequency Domain Processing", processed)

            processed = self.check_inversion(processed)
            self.add_step("Inversion Check", processed)

            result = gray.copy()
            result[y : y + h, x : x + w] = processed
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
                (self.pipeline3, "Pipeline 3"),
            ]:
                logger.info(f"Running {name}...")
                gray_image = (
                    cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                    if len(image.shape) > 2
                    else image.copy()
                )
                result = pipeline.process(gray_image)
                score, metrics = BarcodeImageQuality.evaluate_quality(result)

                # Updated quality evaluation with inversion penalty
                if result is not None:
                    if np.mean(result) > 127:  # Check for inversion
                        score *= 0.7  # Apply penalty, adjust as needed

                logger.info(f"{name} completed with score: {score:.2f}")

                p_result = PipelineResult(
                    name=name,
                    quality_score=score,
                    processed_image=result,
                    steps=pipeline.steps,
                    metrics=metrics,
                )
                self.all_results.append(p_result)

            if not self.all_results:
                logger.warning("No pipeline produced results")
                return image

            self.best_result = max(self.all_results, key=lambda x: x.quality_score)
            logger.info(
                f"Best result: {self.best_result.name} "
                f"(score: {self.best_result.quality_score:.2f})"
            )
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
            rows = 5  # (Row0: Original, Row1: P1, Row2: P2, Row3: P3, Row4: Winner)
            cols = max_steps

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

            # Row 0 => Original
            axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            for c in range(1, cols):
                axes[0, c].axis('off')

            # Rows 1..3 => Pipelines
            for row_index, result in enumerate(self.all_results, start=1):
                # Show steps
                for step_index, (desc, step_img) in enumerate(result.steps):
                    col_index = step_index
                    if col_index < cols:
                        ax = axes[row_index, col_index]
                        ax.imshow(step_img, cmap='gray')
                        ax.set_title(f"Step {step_index+1}: {desc}", fontsize=8)
                        ax.axis('off')

                # Pad remaining columns
                for col_index in range(len(result.steps), cols):
                    axes[row_index, col_index].axis('off')

                # Also show final result in column 0
                ax_final = axes[row_index, 0]
                ax_final.imshow(result.processed_image, cmap='gray')
                ax_final.set_title(f"{result.name} (Final)", fontsize=9)
                ax_final.axis('off')

            # Row 4 => Winner
            if self.best_result:
                axes[4, 0].imshow(self.best_result.processed_image, cmap='gray')
                axes[4, 0].set_title(
                    f"Winner: {self.best_result.name}\n"
                    f"Score: {self.best_result.quality_score:.2f}"
                )
                axes[4, 0].axis('off')

                for c in range(1, cols):
                    axes[4, c].axis('off')
            else:
                for c in range(cols):
                    axes[4, c].axis('off')

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