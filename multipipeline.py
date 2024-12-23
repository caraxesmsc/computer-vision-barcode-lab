import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tkinter import filedialog
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Optional

# -----------------------------------------------------
# Logging Setup
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Data Structures
# -----------------------------------------------------
@dataclass
class PipelineResult:
    """Class to store pipeline results and metrics."""
    name: str
    quality_score: float
    processed_image: np.ndarray
    steps: List[Tuple[str, np.ndarray]]
    metrics: dict

# -----------------------------------------------------
# Barcode Image Quality
# -----------------------------------------------------
class BarcodeImageQuality:
    """Class to evaluate barcode image quality by various metrics."""
    
    @staticmethod
    def calculate_edge_quality(image: np.ndarray) -> float:
        """Combine edge density and Laplacian variance into one score."""
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.mean(edges > 0)
        edge_coherence = cv2.Laplacian(image, cv2.CV_64F).var()
        return edge_density * 100.0 + edge_coherence * 0.1
    
    @staticmethod
    def calculate_contrast_quality(image: np.ndarray) -> float:
        """Compute range between 5th and 95th percentile as a contrast measure."""
        p5, p95 = np.percentile(image, [5, 95])
        return (p95 - p5)
    
    @staticmethod
    def calculate_pattern_quality(image: np.ndarray) -> float:
        """Estimate pattern strength from the vertical Sobel."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        return np.mean(np.abs(sobelx))
    
    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        """Inverse measure of noise: compare image to a blurred version."""
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        noise = np.abs(image.astype(float) - blurred.astype(float))
        # Higher return => less noise
        return 255 - np.mean(noise)
    
    @staticmethod
    def calculate_barcode_area(image: np.ndarray) -> float:
        """
        Return ratio of largest contour area to total image area.
        Helps penalize over-cropping or extremely small barcodes.
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        max_area = max(cv2.contourArea(c) for c in contours)
        total_area = float(image.shape[0] * image.shape[1])
        return max_area / total_area
    
    @staticmethod
    def calculate_inversion_factor(image: np.ndarray) -> float:
        """
        Check if the image is 'inverted' (very high average intensity).
        We'll penalize if average is above ~180 by subtracting some value.
        """
        avg_intensity = np.mean(image)
        if avg_intensity > 180:
            # penalize the image for likely inversion
            return -(avg_intensity - 180.0)
        return 0.0
    
    @classmethod
    def evaluate_quality(cls, image: np.ndarray) -> Tuple[float, dict]:
        """
        Evaluate image based on edge, contrast, pattern, noise, coverage, 
        and whether it's likely inverted. Weighted scoring approach.
        """
        if image is None:
            return 0.0, {}
        
        metrics = {
            'edge_score': cls.calculate_edge_quality(image),
            'contrast_score': cls.calculate_contrast_quality(image),
            'pattern_score': cls.calculate_pattern_quality(image),
            'noise_score': cls.calculate_noise_quality(image),
            'barcode_area': cls.calculate_barcode_area(image),
            'inversion_factor': cls.calculate_inversion_factor(image)
        }
        
        # Weighted sum (tune as needed)
        total_score = (
            metrics['edge_score'] * 0.3 +
            metrics['contrast_score'] * 0.2 +
            metrics['pattern_score'] * 0.2 +
            metrics['noise_score'] * 0.1 +
            metrics['barcode_area'] * 0.3 +         # Encourage bigger barcode coverage
            metrics['inversion_factor'] * 0.1       # Penalize if "inverted"
        )
        
        return total_score, metrics

# -----------------------------------------------------
# Common Utility Functions
# -----------------------------------------------------
def estimate_bar_thickness(image: np.ndarray) -> Tuple[int, int]:
    """
    Estimate barcode bar thickness (min_width, pattern_width).
    We'll do a mild blur, then vertical Sobel, then detect peak diffs.
    """
    try:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        # Use middle portion of the image to avoid noisy borders
        row_slice = sobelx[sobelx.shape[0]//4 : 3*sobelx.shape[0]//4]
        for row in row_slice:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3, 9  # Default fallback

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        # Cap them for safety
        return min(min_width, 10), min(pattern_width, 30)
    except Exception as e:
        logger.error(f"Error in estimate_bar_thickness: {str(e)}")
        return 3, 9

def get_anomalous_regions(anomalies: np.ndarray, bar_thickness: int) -> List[List[int]]:
    """Group contiguous anomalous indices into regions based on bar thickness."""
    regions = []
    if len(anomalies) == 0:
        return regions
    try:
        current_region = [anomalies[0]]
        for i in range(1, len(anomalies)):
            if (anomalies[i] == anomalies[i-1] + 1 and
                anomalies[i] - current_region[0] < bar_thickness * 2):
                current_region.append(anomalies[i])
            else:
                regions.append(current_region)
                current_region = [anomalies[i]]
        regions.append(current_region)
    except Exception as e:
        logger.error(f"Error in get_anomalous_regions: {str(e)}")
    return regions

def detect_barcode_region(image: np.ndarray, min_region_size: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the barcode region by vertical edges. 
    Returns (x,y,w,h) or None.
    """
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
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect_ratio < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None
        
        # Return region with highest edge density
        return max(valid_regions,
                   key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))
    except Exception as e:
        logger.error(f"Error in detect_barcode_region: {str(e)}")
        return None

# -----------------------------------------------------
# Pipeline Base Class
# -----------------------------------------------------
class PipelineBase:
    """Base class for all barcode processing pipelines."""
    def __init__(self):
        self.steps = []
        self.problems = []

    def add_step(self, description: str, image: np.ndarray):
        """Add a processing step to the pipeline's step list."""
        try:
            if image is not None:
                self.steps.append((description, image.copy()))
        except Exception as e:
            logger.error(f"Error adding step {description}: {str(e)}")

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image. Must be implemented by subclasses.
        Returns processed image.
        """
        raise NotImplementedError

    def preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        except Exception as e:
            logger.error(f"Error in preprocess_grayscale: {str(e)}")
            return image.copy()

    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int,int,int,int]]]:
        """
        Detect problems in the image. 
        Must be implemented by subclass.
        """
        raise NotImplementedError

# -----------------------------------------------------
# Pipeline 1: Pattern-Based Approach
# -----------------------------------------------------
class Pipeline1(PipelineBase):
    """
    Pipeline1 focuses on a pattern-based approach to detect foreign objects,
    salt & pepper, etc., then rotate if needed, and finally binarize.
    """
    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int,int,int,int]]]:
        """Detect image problems with improved accuracy for codebase 1 style."""
        problems = []
        region = detect_barcode_region(image)
        if region is None:
            logger.warning("No barcode region detected in Pipeline1")
            return problems, None
        
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]

        # bar thickness
        min_bar_width, pattern_width = estimate_bar_thickness(roi)

        # foreign object detection
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        v_anomalies = np.where(np.abs(vertical_profile - np.mean(vertical_profile)) > 3.0*v_std)[0]
        h_anomalies = np.where(np.abs(horizontal_profile - np.mean(horizontal_profile)) > 3.0*h_std)[0]
        if len(v_anomalies) > min_bar_width or len(h_anomalies) > min_bar_width:
            problems.append('foreign_object')

        # salt & pepper
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        if np.percentile(noise_diff, 95) > 40:
            problems.append('salt_pepper')

        # rotation detection
        edges = cv2.Canny(roi, 100, 200)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                problems.append('rotation')

        return problems, region

    def remove_foreign_object(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        """Remove foreign objects by vertical averaging."""
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()
        min_bar_width, pattern_width = estimate_bar_thickness(roi)

        # vertical kernel
        kernel = np.ones((h, min_bar_width)) / float(h)

        # detect anomalies
        vertical_profile = np.mean(roi, axis=0)
        v_std = np.std(vertical_profile)
        v_anomalous = np.abs(vertical_profile - np.mean(vertical_profile)) > 2.5*v_std

        cleaned = roi.copy()
        for i in range(w):
            if v_anomalous[i]:
                good_cols = []
                for offset in range(-pattern_width, pattern_width+1):
                    check_i = i + offset
                    if 0 <= check_i < w and not v_anomalous[check_i]:
                        good_cols.append(roi[:, check_i])
                if good_cols:
                    cleaned[:, i] = np.mean(good_cols, axis=0)

        # final vertical averaging
        filtered = cv2.filter2D(cleaned, -1, kernel)

        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def remove_salt_pepper(self, image: np.ndarray) -> np.ndarray:
        """Progressive median filter."""
        result = cv2.medianBlur(image, 3)
        result = cv2.medianBlur(result, 5)
        noise_level = np.std(result - image)
        if noise_level > 30:
            result = cv2.medianBlur(result, 3)
        return result

    def fix_rotation(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        """Fix rotation carefully with integer dsize to avoid warpAffine errors."""
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()

        # padding
        pad = int(max(w, h)*0.3)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, 
                                    cv2.BORDER_CONSTANT, value=255)

        edges = cv2.Canny(padded, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        if lines is not None:
            angles = []
            for rho, theta in lines[:,0]:
                angle_deg = np.degrees(theta)
                if angle_deg < 45:
                    angles.append(angle_deg)
                elif angle_deg > 135:
                    angles.append(angle_deg - 180)
            
            if angles:
                median_angle = np.median(angles)
                rows, cols = padded.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1.0)

                # compute new dims
                angle_rad = np.radians(abs(median_angle))
                new_w = int(cols * np.cos(angle_rad) + rows * np.sin(angle_rad))
                new_h = int(rows * np.cos(angle_rad) + cols * np.sin(angle_rad))
                # shift
                M[0,2] += (new_w - cols)/2
                M[1,2] += (new_h - rows)/2

                rotated = cv2.warpAffine(
                    padded, M,
                    (new_w, new_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )

                # find content bounding rect
                edges2 = cv2.Canny(rotated, 50, 150)
                coords = np.column_stack(np.where(edges2 > 0))
                if len(coords) > 0:
                    rx, ry, rw, rh = cv2.boundingRect(coords)
                    # crop
                    rx2 = rx + rw
                    ry2 = ry + rh
                    if rx2>rotated.shape[1]: rx2=rotated.shape[1]
                    if ry2>rotated.shape[0]: ry2=rotated.shape[0]
                    rotated_crop = rotated[ry:ry2, rx:rx2]

                    # resize if needed to fit original roi
                    if rotated_crop.shape != (h, w):
                        rotated_crop = cv2.resize(rotated_crop, (w,h))
                    
                    result = image.copy()
                    result[y:y+h, x:x+w] = rotated_crop
                    return result
        return image

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Binarize using normalized adaptive threshold to avoid holes."""
        p2, p98 = np.percentile(image, (2, 98))
        normed = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            normed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51, 5
        )
        kernel = np.ones((2,2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Main pipeline logic for Pipeline1."""
        self.steps = []
        self.add_step("Original Image", image)

        gray = self.preprocess_grayscale(image)

        problems, region = self.detect_and_log_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if region is None:
            return gray

        processed = gray.copy()

        # handle rotation first
        if 'rotation' in problems:
            processed = self.fix_rotation(processed, region)
            self.add_step("Fixed Rotation", processed)

            # re-detect region
            new_region = detect_barcode_region(processed)
            if new_region:
                region = new_region

        # foreign object
        if 'foreign_object' in problems:
            processed = self.remove_foreign_object(processed, region)
            self.add_step("Removed Foreign Object", processed)

        # salt & pepper
        if 'salt_pepper' in problems:
            processed = self.remove_salt_pepper(processed)
            self.add_step("Removed Salt & Pepper", processed)

        # final binarization
        x,y,w,h = region
        roi = processed[y:y+h, x:x+w]
        binary = self.binarize(roi)
        self.add_step("Binarized", binary)
        processed[y:y+h, x:x+w] = binary

        return processed

# -----------------------------------------------------
# Pipeline 2: Morphological Approach
# -----------------------------------------------------
class Pipeline2(PipelineBase):
    """
    Morphological approach focusing on shape and structure, 
    plus final vertical averaging if 'foreign_object' is detected.
    """

    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int,int,int,int]]]:
        region = detect_barcode_region(image)
        if region is None:
            return [], None
        
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]

        problems = []

        # morphological gradient for structure
        kernel = np.ones((3,3), np.uint8)
        gradient = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel)
        if np.mean(gradient) < 20:
            problems.append('structure')

        # top-hat for noise detection
        tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
        if np.mean(tophat) > 30:
            problems.append('noise')

        # Suppose we also guess 'foreign_object' if some large black region
        if np.mean(roi) < 50:
            problems.append('foreign_object')

        return problems, region

    def _apply_morphological_operations(self, roi: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        # Opening => remove small noise
        opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        # Closing => fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def _enhance_contrast(self, roi: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(roi)

    def _final_vertical_averaging(self, roi: np.ndarray) -> np.ndarray:
        """Vertical averaging pass if foreign object was detected."""
        bar_thickness, _ = estimate_bar_thickness(roi)
        bar_thickness = max(1, bar_thickness)
        kernel = np.ones((roi.shape[0], bar_thickness)) / float(roi.shape[0])
        cleaned = np.zeros_like(roi)
        for i in range(roi.shape[1] - bar_thickness + 1):
            section = roi[:, i:i+bar_thickness]
            avg_col = np.mean(section, axis=1)
            cleaned[:, i:i+bar_thickness] = avg_col.reshape(-1,1)
        return cleaned

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        gray = self.preprocess_grayscale(image)
        problems, region = self.detect_and_log_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if region is None:
            return gray

        x, y, w, h = region
        roi = gray[y:y+h, x:x+w]

        # morphological ops
        morph = self._apply_morphological_operations(roi)
        self.add_step("Morphological Processing", morph)

        # enhance contrast
        enh = self._enhance_contrast(morph)
        self.add_step("Enhanced", enh)

        # if foreign object => final vertical averaging
        if 'foreign_object' in problems:
            final = self._final_vertical_averaging(enh)
            self.add_step("Vertical Avg Cleanup", final)
            enh = final

        result = gray.copy()
        result[y:y+h, x:x+w] = enh
        return result

# -----------------------------------------------------
# Pipeline 3: Frequency-Domain Approach
# -----------------------------------------------------
class Pipeline3(PipelineBase):
    """
    Frequency-domain approach focusing on noise reduction and pattern enhancement.
    """

    def detect_and_log_problems(self, image: np.ndarray) -> Tuple[List[str], Optional[Tuple[int,int,int,int]]]:
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

        # Check for a strong band => 'periodic_noise'
        high_freq = magnitude_spectrum[0:magnitude_spectrum.shape[0]//4, :]
        if np.max(high_freq) > 150:
            problems.append('periodic_noise')

        return problems, region

    def _fix_frequency_domain(self, roi: np.ndarray) -> np.ndarray:
        try:
            f_transform = np.fft.fft2(roi)
            f_shift = np.fft.fftshift(f_transform)
            rows, cols = roi.shape
            crow, ccol = rows//2, cols//2

            mask = np.ones((rows, cols), np.uint8)
            r = 30
            xx, yy = np.ogrid[:rows, :cols]
            mask_area = (xx - crow)**2 + (yy - ccol)**2 <= r*r
            mask[mask_area] = 0

            f_shift *= mask
            f_ishift = np.fft.ifftshift(f_shift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cleaned
        except Exception as e:
            logger.error(f"Error in frequency domain fix: {str(e)}")
            return roi

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        gray = self.preprocess_grayscale(image)
        problems, region = self.detect_and_log_problems(gray)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)

        if region is None:
            return gray

        x, y, w, h = region
        roi = gray[y:y+h, x:x+w].copy()

        processed = roi
        # If any freq domain flags:
        if 'frequency_domain' in problems or 'periodic_noise' in problems:
            processed = self._fix_frequency_domain(roi)
            self.add_step("Frequency Domain Processing", processed)

        result = gray.copy()
        result[y:y+h, x:x+w] = processed
        return result

# -----------------------------------------------------
# Multi-Pipeline Manager
# -----------------------------------------------------
class BarcodeProcessor:
    """Manager class for running and evaluating multiple pipelines, then picking a winner."""
    def __init__(self):
        self.pipeline1 = Pipeline1()
        self.pipeline2 = Pipeline2()
        self.pipeline3 = Pipeline3()
        self.all_results: List[PipelineResult] = []
        self.best_result: Optional[PipelineResult] = None
        self.original_image: Optional[np.ndarray] = None

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run all pipelines, evaluate, and store the best result."""
        try:
            self.original_image = image.copy()
            self.all_results.clear()
            
            for pipeline, name in [
                (self.pipeline1, "Pipeline 1"),
                (self.pipeline2, "Pipeline 2"),
                (self.pipeline3, "Pipeline 3")
            ]:
                try:
                    logger.info(f"Running {name}...")
                    result = pipeline.process(image.copy())
                    score, metrics = BarcodeImageQuality.evaluate_quality(result)
                    logger.info(f"{name} completed with score: {score:.2f}")

                    p_res = PipelineResult(
                        name=name,
                        quality_score=score,
                        processed_image=result,
                        steps=pipeline.steps,
                        metrics=metrics
                    )
                    self.all_results.append(p_res)
                except Exception as e:
                    logger.error(f"Error in {name}: {str(e)}")
                    continue

            if not self.all_results:
                logger.warning("No pipeline produced any result.")
                return image

            self.best_result = max(self.all_results, key=lambda x: x.quality_score)
            logger.info(f"Best result: {self.best_result.name} "
                        f"(score: {self.best_result.quality_score:.2f})")
            
            return self.best_result.processed_image

        except Exception as e:
            logger.error(f"Error in main processing: {str(e)}")
            return image

    def visualize_all_results(self):
        """Create a figure showing original image (row 0) + pipeline steps (rows1..3) + winner (row4)."""
        try:
            if not self.all_results or self.original_image is None:
                logger.warning("No results to visualize")
                return

            # We find the maximum steps among all pipelines
            max_steps = max(len(r.steps) for r in self.all_results)
            rows = 5  # row0=original, row1..3= pipeline steps, row4=winner
            cols = max_steps

            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

            # Row 0 => Original
            axes[0,0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            axes[0,0].set_title("Original Image", fontsize=8)
            axes[0,0].axis('off')
            for c in range(1, cols):
                axes[0,c].axis('off')

            # Rows 1..3 => pipeline results
            for row_index, result in enumerate(self.all_results, start=1):
                # pipeline final result in col 0
                final_img = result.processed_image
                axes[row_index, 0].imshow(final_img, cmap='gray')
                axes[row_index, 0].set_title(result.name + " (Final)", fontsize=8)
                axes[row_index, 0].axis('off')

                # Then each step in subsequent cols
                for step_index, (desc, img) in enumerate(result.steps, start=1):
                    if step_index >= cols:
                        break
                    ax = axes[row_index, step_index]
                    ax.imshow(img, cmap='gray')
                    ax.set_title(desc, fontsize=8)
                    ax.axis('off')

                # hide extra columns if any
                for extra_col in range(len(result.steps)+1, cols):
                    axes[row_index, extra_col].axis('off')

            # Row 4 => Winner
            if self.best_result is not None:
                winner_img = self.best_result.processed_image
                axes[4, 0].imshow(winner_img, cmap='gray')
                axes[4, 0].set_title(f"Winner: {self.best_result.name}", fontsize=8)
                axes[4, 0].axis('off')
                
                # show metrics in col1
                m = self.best_result.metrics
                metrics_text = (
                    f"Edge Score: {m.get('edge_score',0):.2f}\n"
                    f"Contrast Score: {m.get('contrast_score',0):.2f}\n"
                    f"Pattern Score: {m.get('pattern_score',0):.2f}\n"
                    f"Noise Score: {m.get('noise_score',0):.2f}\n"
                    f"BarcodeArea: {m.get('barcode_area',0):.2f}\n"
                    f"InversionFactor: {m.get('inversion_factor',0):.2f}\n"
                    f"Total: {self.best_result.quality_score:.2f}"
                )
                axes[4,1].text(0.05, 0.5, metrics_text, fontsize=10, va='center')
                axes[4,1].axis('off')
                # hide leftover cols
                for c in range(2, cols):
                    axes[4,c].axis('off')
            else:
                # no winner
                for c in range(cols):
                    axes[4,c].axis('off')

            plt.tight_layout()
            plt.savefig("pipeline_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Results saved to pipeline_comparison.png")
        except Exception as e:
            logger.error(f"Error in visualize_all_results: {str(e)}")

# -----------------------------------------------------
# Main Execution
# -----------------------------------------------------
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
