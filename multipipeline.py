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
    # -------------------------------------
# Updated Winner Selection Logic
# -------------------------------------
    def evaluate_quality(self, image: np.ndarray) -> Tuple[float, dict]:
        """Enhanced quality metrics for better winner selection."""
        if image is None:
            return 0.0, {}

        # Edge quality with improved pattern detection
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.mean(edges > 0)
        edge_coherence = cv2.Laplacian(image, cv2.CV_64F).var()
        edge_score = edge_density * 100 + edge_coherence * 0.1

        # Contrast quality that better handles dark images
        p5, p95 = np.percentile(image, [5, 95])
        mean_intensity = np.mean(image)
        contrast_score = p95 - p5
        
        # Don't overly penalize dark/bright images
        if 10 < mean_intensity < 245:  
            contrast_score *= 1.2

        # Pattern quality focusing on barcode structure
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        pattern_score = np.mean(np.abs(sobelx)) / (np.mean(np.abs(sobely)) + 1e-6)

        # Noise quality
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        noise = np.abs(image.astype(float) - blurred.astype(float))
        noise_score = 255 - np.mean(noise)

        # Salt & pepper specific detection
        median = cv2.medianBlur(image, 3)
        salt_pepper_diff = np.abs(image - median)
        has_salt_pepper = np.percentile(salt_pepper_diff, 95) > 30

        metrics = {
            'edge_score': edge_score,
            'contrast_score': contrast_score,
            'pattern_score': pattern_score,
            'noise_score': noise_score,
            'has_salt_pepper': has_salt_pepper
        }

        # Weighted score calculation
        total_score = (
            edge_score * 0.4 +
            contrast_score * 0.3 +
            pattern_score * 0.2 +
            noise_score * 0.1
        )

        # Bonuses for desired characteristics
        if pattern_score > 2.0:  # Strong vertical pattern
            total_score *= 1.2
        if not has_salt_pepper:  # Clean image
            total_score *= 1.1
        if 127 > mean_intensity > 30:  # Proper black text on white background
            total_score *= 1.15

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

        kernel = np.ones((h, min_bar_width)) /h
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

    def process_salt_pepper(self, image, barcode_region):
        """Enhanced salt & pepper handling with projection-based cropping."""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Multi-stage median filtering
        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)
        
        # Additional filtering if needed
        noise_level = np.std(filtered - roi)
        if noise_level > 20:
            filtered = cv2.medianBlur(filtered, 3)

        # Apply projection-based cropping
        binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        vertical_projection = np.sum(binary == 0, axis=0)
        horizontal_projection = np.sum(binary == 0, axis=1)

        # Find valid barcode region
        v_threshold = 0.6 * np.max(vertical_projection)
        h_threshold = 0.6 * np.max(horizontal_projection)
        
        v_indices = np.where(vertical_projection > v_threshold)[0]
        h_indices = np.where(horizontal_projection > h_threshold)[0]

        if len(v_indices) > 0 and len(h_indices) > 0:
            x_start, x_end = v_indices[0], v_indices[-1]
            y_start, y_end = h_indices[0], h_indices[-1]
            
            # Add padding
            pad_x = int(0.05 * (x_end - x_start))
            pad_y = int(0.05 * (y_end - y_start))
            
            x_start = max(0, x_start - pad_x)
            x_end = min(w, x_end + pad_x)
            y_start = max(0, y_start - pad_y)
            y_end = min(h, y_end + pad_y)
            
            filtered = filtered[y_start:y_end, x_start:x_end]

        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def fix_rotation(self, image, barcode_region):
        """Enhanced rotation correction using Mark's code approach."""
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()

        # Add padding before rotation
        pad = int(max(w, h) * 0.3)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, 
                                cv2.BORDER_CONSTANT, value=255)

        # Enhanced edge detection
        edges = cv2.Canny(padded, 50, 150)
        
        # More precise line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
                elif 45 <= angle <= 135:
                    angles.append(angle - 90)

            if angles:
                # Weighted median for better angle estimation
                sorted_angles = sorted(angles)
                weights = [1/(i+1) for i in range(len(sorted_angles))]
                median_angle = np.average(sorted_angles, weights=weights)
                
                rows, cols = padded.shape
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                
                # Calculate new dimensions to avoid cropping
                angle_rad = abs(np.radians(median_angle))
                new_w = int(cols * abs(np.cos(angle_rad)) + rows * abs(np.sin(angle_rad)))
                new_h = int(rows * abs(np.cos(angle_rad)) + cols * abs(np.sin(angle_rad)))
                
                # Adjust transformation matrix
                matrix[0, 2] += (new_w - cols)/2
                matrix[1, 2] += (new_h - rows)/2

                # Apply rotation with border replication
                rotated = cv2.warpAffine(padded, matrix, (new_w, new_h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)

                # Find the actual barcode region in rotated image
                edges = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges > 0))
                if len(coords) > 0:
                    x_, y_, w_, h_ = cv2.boundingRect(coords)
                    rotated_crop = rotated[y_:y_+h_, x_:x_+w_]
                    
                    # Ensure output size matches input
                    if rotated_crop.shape != (h, w):
                        rotated_crop = cv2.resize(rotated_crop, (w, h))
                    
                    result = image.copy()
                    result[y:y+h, x:x+w] = rotated_crop
                    return result

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

    def final_cleanup(self, binary):
        """More conservative final cleanup that preserves thin bars."""
        min_bar_width, _ = self.estimate_bar_thickness(binary)
        min_bar_width = max(1, min(min_bar_width, 5))  # Limit max width
        
        kernel = np.ones((binary.shape[0], min_bar_width)) / binary.shape[0]
        cleaned = np.zeros_like(binary)
        
        for i in range(binary.shape[1] - min_bar_width + 1):
            region = binary[:, i:i+min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            # Higher threshold to prevent black image
            cleaned[:, i:i+min_bar_width] = (vertical_avg > 150).astype(np.uint8) * 255
            
        return cleaned
    
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
#-------------------------------------------------------------------
#Pipeline 2: Morphological Approach + Pipeline 1 Improvements
#-------------------------------------------------------------------
"""Updates for Pipeline2 and Pipeline3."""

# -------------------------------------
# Updated Pipeline2
# -------------------------------------
class Pipeline2(PipelineBase):
    """Morphological approach with improved contrast handling."""
    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
        try:
            region = detect_barcode_region(image)
            if region is None:
                return [], None

            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            problems = []

            # Structure analysis using gradients
            kernel = np.ones((3, 3), np.uint8)
            gradient = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel)
            if np.mean(gradient) < 20:
                problems.append('structure')

            # Noise detection
            tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
            if np.mean(tophat) > 30:
                problems.append('noise')

            # Additional problem detection
            p5, p95 = np.percentile(roi, [5, 95])
            if (p95 - p5) < 80:
                problems.append('low_contrast')

            laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
            if laplacian_var < 100:
                problems.append('blur')

            return problems, region

        except Exception as e:
            logger.error(f"Error in Pipeline2 problem detection: {str(e)}")
            return [], None

    def _process_morphology(self, image: np.ndarray) -> np.ndarray:
        """Enhanced morphological operations."""
        try:
            kernel = np.ones((3, 3), np.uint8)
            
            # First stage: Remove noise
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # Second stage: Fill small gaps
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # Third stage: Enhance vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            vertical_enhanced = cv2.morphologyEx(closed, cv2.MORPH_DILATE, vertical_kernel)
            
            return vertical_enhanced
        except Exception as e:
            logger.error(f"Error in morphological operations: {str(e)}")
            return image

    def fix_blur(self, image: np.ndarray) -> np.ndarray:
        """Gentler blur correction to preserve thin lines."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 0.5)  # Reduced sigma
        unsharp = cv2.addWeighted(image, 1.3, gaussian, -0.3, 0)  # Gentler weights
        return unsharp

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        try:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduced clip limit
            enhanced = clahe.apply(image)
            
            # Adaptive gamma correction
            mean_intensity = np.mean(enhanced)
            if mean_intensity < 128:
                gamma = 1.1  # Slightly brighten dark images
            else:
                gamma = 0.9  # Slightly darken bright images
            inv_gamma = 1.0 / gamma
            table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)]).astype(np.uint8)
            gamma_corrected = cv2.LUT(enhanced, table)
            
            return gamma_corrected
        except Exception as e:
            logger.error(f"Error in contrast enhancement: {str(e)}")
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
            roi = gray[y:y+h, x:x+w]

            # Process in optimal order
            processed = self._process_morphology(roi)
            self.add_step("Morphological Processing", processed)

            if 'blur' in probs:
                processed = self.fix_blur(processed)
                self.add_step("Sharpened", processed)

            if 'low_contrast' in probs:
                processed = self._enhance_contrast(processed)
                self.add_step("Enhanced Contrast", processed)

            # Check and correct inversion
            if np.mean(processed) > 127:
                processed = 255 - processed
                self.add_step("Inversion Corrected", processed)

            result = gray.copy()
            result[y:y+h, x:x+w] = processed
            return result

        except Exception as e:
            logger.error(f"Error in Pipeline2 process: {str(e)}")
            return image

# -------------------------------------
# Updated Pipeline3
# -------------------------------------
class Pipeline3(PipelineBase):
    """Frequency domain approach with improved pattern enhancement."""
    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
        try:
            region = detect_barcode_region(image)
            if region is None:
                return [], None

            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            problems = []

            # Frequency domain analysis
            f_transform = np.fft.fft2(roi)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

            # Look for high-frequency noise
            high_freq = magnitude_spectrum[magnitude_spectrum.shape[0]//4:3*magnitude_spectrum.shape[0]//4,
                                        magnitude_spectrum.shape[1]//4:3*magnitude_spectrum.shape[1]//4]
            if np.std(high_freq) > 45:
                problems.append('frequency_domain')

            # Check for periodic noise
            periodic_pattern = np.sum(magnitude_spectrum > 150)
            if periodic_pattern > magnitude_spectrum.size * 0.01:
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

            # Create butterworth high-pass filter
            d0 = 30  # Cutoff frequency
            n = 2    # Order of the filter
            u = np.arange(rows)
            v = np.arange(cols)
            u, v = np.meshgrid(u-crow, v-ccol)
            d = np.sqrt(u*u + v*v)
            h = 1 / (1 + (d0/d)**(2*n))
            h = np.fft.fftshift(h)

            # Apply filter
            f_shift_filtered = f_shift * h
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)

            # Normalize and enhance contrast
            img_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(img_norm)

        except Exception as e:
            logger.error(f"Error in frequency domain fix: {str(e)}")
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
            roi = gray[y:y+h, x:x+w]

            processed = roi.copy()
            if 'frequency_domain' in probs or 'periodic_noise' in probs:
                processed = self._fix_frequency_domain(roi)
                self.add_step("Frequency Domain Processing", processed)

            # Check and correct inversion
            if np.mean(processed) > 127:
                processed = 255 - processed
                self.add_step("Inversion Corrected", processed)

            result = gray.copy()
            result[y:y+h, x:x+w] = processed
            return result

        except Exception as e:
            logger.error(f"Error in Pipeline3 process: {str(e)}")
            return image
#-------------------------------------------------------------------
#Multi-Pipeline Manager
#-------------------------------------------------------------------
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

#-------------------------------------------------------------------
#Main Execution
#-------------------------------------------------------------------
"""Final Updates: BarcodeProcessor Manager and Main."""

# -------------------------------------
# Updated BarcodeProcessor Manager
# -------------------------------------
class BarcodeProcessor:
    """Enhanced manager class for running and evaluating multiple pipelines."""

    def __init__(self):
        self.pipeline1 = Pipeline1()
        self.pipeline2 = Pipeline2()
        self.pipeline3 = Pipeline3()
        self.all_results: List[PipelineResult] = []
        self.best_result: Optional[PipelineResult] = None
        self.original_image: Optional[np.ndarray] = None

    def evaluate_pipeline_result(self, result: np.ndarray, name: str) -> Tuple[float, dict]:
        """Enhanced pipeline result evaluation."""
        if result is None:
            return 0.0, {}

        # Edge quality (40%)
        edges = cv2.Canny(result, 50, 150)
        edge_density = np.mean(edges > 0)
        edge_coherence = cv2.Laplacian(result, cv2.CV_64F).var()
        edge_score = edge_density * 100 + edge_coherence * 0.1

        # Contrast quality (30%)
        p5, p95 = np.percentile(result, [5, 95])
        mean_intensity = np.mean(result)
        contrast_score = p95 - p5

        # Pattern quality (20%)
        sobelx = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=3)
        pattern_score = np.mean(np.abs(sobelx)) / (np.mean(np.abs(sobely)) + 1e-6)

        # Noise quality (10%)
        blurred = cv2.GaussianBlur(result, (3, 3), 0)
        noise = np.abs(result.astype(float) - blurred.astype(float))
        noise_score = 255 - np.mean(noise)

        # Calculate base score
        metrics = {
            'edge_score': edge_score,
            'contrast_score': contrast_score,
            'pattern_score': pattern_score,
            'noise_score': noise_score,
            'mean_intensity': mean_intensity
        }

        # Base score calculation
        score = (
            edge_score * 0.4 +
            contrast_score * 0.3 +
            pattern_score * 0.2 +
            noise_score * 0.1
        )

        # Apply bonuses/penalties based on known issues
        if pattern_score > 2.0:  # Strong barcode pattern
            score *= 1.2
            metrics['pattern_bonus'] = True

        if 30 < mean_intensity < 225:  # Not too dark or bright
            score *= 1.1
            metrics['intensity_bonus'] = True

        if name == "Pipeline 1" and 'salt_pepper' in self.pipeline1.problems:
            if np.std(result) > 20:  # Still has good contrast
                score *= 1.15  # Bonus for Pipeline 1 handling salt & pepper
                metrics['salt_pepper_bonus'] = True

        if mean_intensity < 30 and contrast_score > 100:  # Dark but still readable
            score *= 1.2
            metrics['dark_contrast_bonus'] = True

        return score, metrics

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image through all pipelines with improved selection."""
        try:
            self.original_image = image.copy()
            self.all_results = []
            logger.info("Starting barcode processing...")

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
                score, metrics = self.evaluate_pipeline_result(result, name)
                logger.info(f"{name} completed with score: {score:.2f}")

                # Create result object
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

            # Enhanced winner selection logic
            weighted_results = []
            for result in self.all_results:
                weight = 1.0
                metrics = result.metrics
                
                # Additional weighting factors
                if metrics.get('pattern_bonus'):
                    weight *= 1.2
                if metrics.get('intensity_bonus'):
                    weight *= 1.1
                if metrics.get('salt_pepper_bonus'):
                    weight *= 1.15
                if metrics.get('dark_contrast_bonus'):
                    weight *= 1.2
                
                weighted_results.append((result, result.quality_score * weight))

            # Select best result based on weighted scores
            self.best_result = max(weighted_results, key=lambda x: x[1])[0]
            
            logger.info(
                f"Best result: {self.best_result.name} "
                f"(score: {self.best_result.quality_score:.2f})"
            )
            return self.best_result.processed_image

        except Exception as e:
            logger.error(f"Error in main processing: {str(e)}")
            return image

    def visualize_all_results(self):
        """Enhanced visualization of pipeline results."""
        try:
            if not self.all_results or self.original_image is None:
                logger.warning("No results to visualize")
                return

            # Setup plot
            max_steps = max(len(r.steps) for r in self.all_results)
            rows = 5  # Original, P1, P2, P3, Winner
            cols = max_steps
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

            # Row 0: Original
            axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            for c in range(1, cols):
                axes[0, c].axis('off')

            # Rows 1-3: Pipeline results
            for row_idx, result in enumerate(self.all_results, start=1):
                # Show processing steps
                for step_idx, (desc, step_img) in enumerate(result.steps):
                    if step_idx < cols:
                        ax = axes[row_idx, step_idx]
                        ax.imshow(step_img, cmap='gray')
                        ax.set_title(f"{desc}\nScore: {result.metrics.get(desc.lower().replace(' ','_')+'_score', ''):.1f}",
                                   fontsize=8)
                        ax.axis('off')

                # Hide unused columns
                for col_idx in range(len(result.steps), cols):
                    axes[row_idx, col_idx].axis('off')

            # Row 4: Winner with metrics
            if self.best_result:
                axes[4, 0].imshow(self.best_result.processed_image, cmap='gray')
                winner_title = (
                    f"Winner: {self.best_result.name}\n"
                    f"Score: {self.best_result.quality_score:.2f}\n"
                    f"Edge: {self.best_result.metrics['edge_score']:.1f}, "
                    f"Contrast: {self.best_result.metrics['contrast_score']:.1f}"
                )
                axes[4, 0].set_title(winner_title, fontsize=9)
                axes[4, 0].axis('off')

                for c in range(1, cols):
                    axes[4, c].axis('off')

            plt.tight_layout()
            plt.savefig("pipeline_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Results visualization saved as 'pipeline_comparison.png'")

        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")

# -------------------------------------
# Main Execution
# -------------------------------------
if __name__ == "__main__":
    try:
        processor = BarcodeProcessor()
        image_path = filedialog.askopenfilename(
            title="Select a Barcode Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if not image_path:
            print("No image selected. Exiting...")
            exit(0)
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            exit(1)
            
        print("Processing image...")
        processed_image = processor.process(image)
        processor.visualize_all_results()
        
        # Save the final processed image
        output_path = "processed_barcode.png"
        cv2.imwrite(output_path, processed_image)
        print(f"\nProcessing complete!")
        print(f"Results visualization saved as 'pipeline_comparison.png'")
        print(f"Processed barcode saved as '{output_path}'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logger.error(f"Error in main execution: {str(e)}")