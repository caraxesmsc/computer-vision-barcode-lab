import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt

from tkinter import filedialog
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple

# -------------------------------------------------------------------
# Data Structures / Utility Classes
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
        # Weighted combination of density & coherence
        return edge_density * 100 + edge_coherence * 0.1

    @staticmethod
    def calculate_contrast_quality(image: np.ndarray) -> float:
        p5, p95 = np.percentile(image, [5, 95])
        contrast_range = p95 - p5
        return contrast_range

    @staticmethod
    def calculate_pattern_quality(image: np.ndarray) -> float:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        pattern_strength = np.mean(np.abs(sobelx))
        return pattern_strength

    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        noise = np.abs(image.astype(float) - blurred.astype(float))
        # The higher the noise, the lower the score => invert
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

        # Weighted total
        total_score = (metrics['edge_score'] * 0.4 +
                       metrics['contrast_score'] * 0.3 +
                       metrics['pattern_score'] * 0.2 +
                       metrics['noise_score'] * 0.1)

        return total_score, metrics

# -------------------------------------------------------------------
# Pipeline 1: Simple Approach (Similar to codebase 3)
# -------------------------------------------------------------------

class GeneralPipeline1:
    """Pipeline 1: Based on codebase 3's approach."""
    def __init__(self, min_region_size: int = 10, min_edge_density: float = 0.1):
        self.steps = []
        self.min_region_size = min_region_size
        self.min_edge_density = min_edge_density

    def add_step(self, description: str, image: np.ndarray):
        self.steps.append((description, image.copy()))

    def _detect_barcode_region(self, image: np.ndarray):
        """Detect barcode region using Sobel edge detection & adaptive thresholding."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        thresh = cv2.adaptiveThreshold(
            gradx, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_region_size or h < self.min_region_size:
                continue

            aspect_ratio = w / float(h)
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi > 50)/(w*h)
            if 1.5 < aspect_ratio < 5 and edge_density > self.min_edge_density:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None

        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        denoised = cv2.medianBlur(image, 3)
        self.add_step("Denoised", denoised)
        return denoised

    def _crop_barcode_region(self, image: np.ndarray) -> np.ndarray:
        region = self._detect_barcode_region(image)
        if region is None:
            print("Barcode region not detected in Pipeline 1.")
            return image
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        self.add_step("ROI", roi)
        return roi

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        self.add_step("Enhanced Contrast", enhanced)
        return enhanced

    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.add_step("Binarized", binary)
        return binary

    def _clean_barcode(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.add_step("Cleaned", cleaned)
        return cleaned

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image.copy()
        denoised = self._denoise_image(gray)
        roi = self._crop_barcode_region(denoised)
        enhanced = self._enhance_contrast(roi)
        binarized = self._binarize_image(enhanced)
        cleaned = self._clean_barcode(binarized)

        return cleaned

# -------------------------------------------------------------------
# Pipeline 2: Morphological Approach (Similar to codebase 6)
# -------------------------------------------------------------------

class GeneralPipeline2:
    """Pipeline 2: Based on codebase 6's morphological approach."""
    def __init__(self, min_region_size: int = 10, min_edge_density: float = 0.1):
        self.steps = []
        self.min_region_size = min_region_size
        self.min_edge_density = min_edge_density

    def add_step(self, description: str, image: np.ndarray):
        self.steps.append((description, image.copy()))

    def _detect_barcode_region(self, image: np.ndarray):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < self.min_region_size or h < self.min_region_size:
                continue
            aspect_ratio = w/float(h)
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi>50)/(w*h)
            if 1.5 < aspect_ratio < 5 and edge_density>self.min_edge_density:
                valid_regions.append((x,y,w,h))

        if not valid_regions:
            return None

        return max(valid_regions, key=lambda r: np.sum(
            gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        ))

    def _crop_barcode_region(self, image: np.ndarray) -> np.ndarray:
        region = self._detect_barcode_region(image)
        if region is None:
            print("Barcode region not detected in Pipeline 2.")
            return image
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        self.add_step("ROI", roi)
        return roi

    def _apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        self.add_step("Morphological Opening", opening)
        return opening

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        self.add_step("CLAHE Enhanced", enhanced)
        return enhanced

    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.add_step("Binarized", binary)
        return binary

    def process(self, image: np.ndarray) -> np.ndarray:
        self.steps = []
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image.copy()
        roi = self._crop_barcode_region(gray)
        opened = self._apply_morphological_operations(roi)
        enhanced = self._enhance_contrast(opened)
        binarized = self._binarize_image(enhanced)

        return binarized

# -------------------------------------------------------------------
# Pipeline 3: Quality-based (like codebase 21 w/ placeholders filled)
# -------------------------------------------------------------------

class GeneralPipeline3:
    """Pipeline 3: Based on codebase 21's approach."""
    def __init__(self, min_region_size: int = 10, min_edge_density: float = 0.1):
        self.steps = []
        self.problems = []
        self.min_region_size = min_region_size
        self.min_edge_density = min_edge_density

    def add_step(self, description: str, image: np.ndarray):
        self.steps.append((description, image.copy()))

    # ~~~~~ DETECT BARCODE REGION ~~~~~
    def _detect_barcode_region(self, image: np.ndarray):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)

        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_region_size or h<self.min_region_size:
                continue

            aspect_ratio = w/float(h)
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi>50)/(w*h)

            if 1.5<aspect_ratio<5 and edge_density>self.min_edge_density:
                valid_regions.append((x,y,w,h))

        if not valid_regions:
            return None

        return max(valid_regions, key=lambda r: np.sum(
            gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        ))

    def _estimate_bar_thickness(self, image: np.ndarray) -> Tuple[int,int]:
        normalized = cv2.normalize(image, None, 0,255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3,3), 0)

        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        midrows = sobelx.shape[0]//4
        for row in sobelx[midrows:3*midrows]:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks)>1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3,9

        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        return min_width, pattern_width

    def _get_anomalous_regions(self, anomalies: np.ndarray, min_bar_width:int):
        """Group contiguous anomalies based on bar thickness."""
        regions = []
        if len(anomalies)==0:
            return regions
        current_region = [anomalies[0]]
        for i in range(1, len(anomalies)):
            if anomalies[i] == anomalies[i-1]+1 and anomalies[i]-current_region[0]<min_bar_width*2:
                current_region.append(anomalies[i])
            else:
                regions.append(current_region)
                current_region=[anomalies[i]]
        regions.append(current_region)
        return regions

    def _detect_salt_pepper(self, roi: np.ndarray)->bool:
        median = cv2.medianBlur(roi,3)
        diff = np.abs(roi - median)
        return np.percentile(diff,95)>40

    def _detect_compression_artifacts(self, gray: np.ndarray)->bool:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0,ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1,ksize=3)
        magnitude = np.sqrt(sobelx**2+sobely**2)

        block_size=8
        blurred = cv2.boxFilter(magnitude.astype(np.float32), -1, (block_size, block_size))
        block_variation = np.std(blurred)
        return block_variation>10

    def _detect_image_problems(self, image: np.ndarray) -> Tuple[List[str], Tuple[int,int,int,int]]:
        problems=[]
        gray = image.copy()
        region = self._detect_barcode_region(gray)
        if region is None:
            return problems,(0,0,gray.shape[1],gray.shape[0])
        x, y, w, h = region

        roi = gray[y:y+h, x:x+w]
        min_bar_width, pattern_width = self._estimate_bar_thickness(roi)

        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)

        v_anomalies = np.where(np.abs(vertical_profile - np.mean(vertical_profile))>3.0*v_std)[0]
        h_anomalies = np.where(np.abs(horizontal_profile - np.mean(horizontal_profile))>3.0*h_std)[0]
        v_anomalous_regions = self._get_anomalous_regions(v_anomalies, min_bar_width)
        h_anomalous_regions = self._get_anomalous_regions(h_anomalies, min_bar_width)
        if len(v_anomalous_regions)>0 or len(h_anomalous_regions)>0:
            problems.append('foreign_object')

        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50,150)
        edge_density = np.mean(edges>0)
        if laplacian_var<90 and edge_density<0.15:
            problems.append('blur')

        p5, p95 = np.percentile(roi,[5,95])
        if (p95 - p5)<80:
            problems.append('low_contrast')

        if self._detect_salt_pepper(roi):
            problems.append('salt_pepper')

        if self._detect_compression_artifacts(roi):
            problems.append('compression_artifacts')

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20*np.log(np.abs(f_shift)+1)
        if np.std(magnitude_spectrum)>45:
            problems.append('frequency_domain')

        lines = cv2.HoughLines(edges,1, np.pi/180, threshold=min(w,h)//3)
        if lines is not None:
            angles=[]
            for _,theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle<45:
                    angles.append(angle)
                elif angle>135:
                    angles.append(angle-180)
            if angles and abs(np.median(angles))>5:
                problems.append('rotation')

        self.problems=problems
        return problems,(x,y,w,h)

    def _remove_salt_pepper(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()
        filtered = cv2.medianBlur(roi, 3)
        result = image.copy()
        result[y:y+h, x:x+w] = filtered
        return result

    def _fix_blur(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()

        gaussian = cv2.GaussianBlur(roi, (0, 0), 2)
        unsharp = cv2.addWeighted(roi, 1.5, gaussian, -0.5, 0)

        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        lap_var = cv2.Laplacian(unsharp, cv2.CV_64F).var()
        if lap_var<150:
            sharpened = cv2.filter2D(unsharp, -1, kernel)
        else:
            sharpened=unsharp

        result = image.copy()
        result[y:y+h, x:x+w] = sharpened
        return result

    def _fix_contrast(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)

        gamma=1.5
        invGamma=1.0/gamma
        table = np.array([((i/255.0)**invGamma)*255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        result = image.copy()
        result[y:y+h, x:x+w] = enhanced
        return result

    def _fix_compression_artifacts(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()

        denoised = cv2.fastNlMeansDenoising(roi, None, h=5, templateWindowSize=7, searchWindowSize=21)
        result = image.copy()
        result[y:y+h, x:x+w] = denoised
        return result

    def _fix_frequency_domain(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = roi.shape
        crow, ccol = rows//2, cols//2
        xx = np.linspace(-ccol, ccol-1, cols)
        yy = np.linspace(-crow, crow-1, rows)
        xv, yv = np.meshgrid(xx, yy)
        sigma=30
        gauss = np.exp(-(xv**2 + yv**2)/(2*sigma**2))
        mask = 1-gauss

        f_shift = f_shift*mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def _remove_foreign_object(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()

        min_bar_width, pattern_width = self._estimate_bar_thickness(roi)
        v_kernel = np.ones((h, min_bar_width))/h
        h_kernel = np.ones((min_bar_width, w))/w

        vertical_avg = cv2.filter2D(roi, -1, v_kernel)
        horizontal_avg = cv2.filter2D(roi, -1, h_kernel)

        v_std = np.std(vertical_avg)
        h_std = np.std(horizontal_avg)
        v_anomalous = np.abs(roi - vertical_avg)>2.5*v_std
        h_anomalous = np.abs(roi - horizontal_avg)>2.5*h_std

        cleaned=roi.copy()
        # Replace anomalous columns
        for i in range(w):
            if np.any(v_anomalous[:, i]):
                good_cols=[]
                for j in range(max(0, i-pattern_width), min(w, i+pattern_width+1)):
                    if not np.any(v_anomalous[:, j]):
                        good_cols.append(roi[:, j])
                if good_cols:
                    cleaned[:, i]=np.mean(good_cols, axis=0)

        # Replace anomalous rows
        for j in range(h):
            if np.any(h_anomalous[j,:]):
                good_rows=[]
                for i in range(max(0, j-pattern_width), min(h, j+pattern_width+1)):
                    if not np.any(h_anomalous[i,:]):
                        good_rows.append(cleaned[i,:])
                if good_rows:
                    cleaned[j,:]=np.mean(good_rows, axis=0)

        result = image.copy()
        result[y:y+h, x:x+w] = cleaned
        return result

    def _fix_rotation(self, image: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w].copy()

        pad=int(max(w,h)*0.3)
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

        edges = cv2.Canny(padded, 50, 150)
        lines = cv2.HoughLines(edges,1, np.pi/180, threshold=50)
        if lines is not None:
            angles=[]
            for rho, theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle<45:
                    angles.append(angle)
                elif angle>135:
                    angles.append(angle-180)
            if angles:
                median_angle = np.median(angles)
                rows, cols = padded.shape
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)

                angle_rad=np.radians(abs(median_angle))
                new_width=int(cols*np.abs(np.cos(angle_rad)) + rows*np.abs(np.sin(angle_rad)))
                new_height=int(rows*np.abs(np.cos(angle_rad))+cols*np.abs(np.sin(angle_rad)))
                matrix[0,2]+= (new_width-cols)/2
                matrix[1,2]+= (new_height-rows)/2

                rotated = cv2.warpAffine(padded, matrix, (new_width, new_height),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255,255,255))
                edges2 = cv2.Canny(rotated, 50,150)
                coords = np.column_stack(np.where(edges2>0))
                if len(coords)>0:
                    x_,y_,w_,h_ = cv2.boundingRect(coords)
                    x_ = max(0, min(x_, rotated.shape[1]-w_))
                    y_ = max(0, min(y_, rotated.shape[0]-h_))
                    rotated_crop = rotated[y_:y_+h_, x_:x_+w_]

                    if rotated_crop.shape[0]>0.5*h and rotated_crop.shape[1]>0.5*w:
                        result = image.copy()
                        result[y:y+h, x:x+w] = rotated_crop
                        return result

        return image

    def _binarize(self, image: np.ndarray)->np.ndarray:
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.interp(image, (p2, p98), (0,255)).astype(np.uint8)

        binary = cv2.adaptiveThreshold(
            normalized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51,5
        )
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _validate_barcode_elements(self, binary: np.ndarray)->np.ndarray:
        min_bar_width, _ = self._estimate_bar_thickness(binary)

        num_labels, labels = cv2.connectedComponents(binary)
        unique_labels, counts = np.unique(labels, return_counts=True)

        for label, count in zip(unique_labels, counts):
            if label!=0 and count< min_bar_width*5:
                binary[labels==label]=0

        # Optionally remove weird wide elements
        for label in unique_labels:
            if label!=0:
                element = (labels==label)
                if np.sum(element)>0:
                    x, y, w, h = cv2.boundingRect(element.astype(np.uint8))
                    if w< min_bar_width*0.5 or w>min_bar_width*5:
                        binary[element]=0

        return binary

    def _final_barcode_cleanup(self, binary: np.ndarray)->np.ndarray:
        min_bar_width, _ = self._estimate_bar_thickness(binary)
        min_bar_width = max(1, int(min_bar_width))
        v_kernel = np.ones((binary.shape[0], min_bar_width))/binary.shape[0]

        cleaned = np.zeros_like(binary)
        for i in range(0, binary.shape[1]-min_bar_width+1):
            region = binary[:, i:i+min_bar_width]
            vertical_avg = np.mean(region, axis=0)
            cleaned[:, i:i+min_bar_width] = (vertical_avg>127).astype(np.uint8)*255

        return cleaned

    def _determine_barcode_symbology(self, binary: np.ndarray)->str:
        """Try to guess the symbology by analyzing patterns of black/white runs."""
        min_bar_width,_ = self._estimate_bar_thickness(binary)
        elements = np.sum(binary==0, axis=0)//(min_bar_width if min_bar_width>0 else 1)

        if elements.shape[0]<4:
            return "Unknown"

        # Some naive pattern checks
        if (np.all(elements[:4]==[1,1,1,1]) and
            np.all(elements[-3:]==[2,1,1])):
            return "EAN-13"
        elif (np.all(elements[:3]==[1,1,1]) and
              np.all(elements[-3:]==[1,1,1])):
            return "EAN-8"
        elif (np.all(elements[:4]==[1,1,2,1])):
            return "UPC-A"
        elif (np.all(elements[:4]==[1,1,1])):
            return "UPC-E"
        elif (np.all(elements[:3]==[2,1,1]) and
              np.all(elements[-3:]==[2,1,1])):
            return "Code 39"
        elif (np.all(elements[:4]==[1,1,2,1]) and 
              np.all(elements[-3:]==[2,1,1])):
            return "Code 128"
        else:
            return "Unknown"

    def preprocess_image(self, image: np.ndarray)->np.ndarray:
        self.steps=[]
        self.add_step("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image.copy()
        problems, region = self._detect_image_problems(gray)
        self.add_step("Detected Problems: "+", ".join(problems), gray)

        if region is None:
            print("Barcode region not found.")
            return gray

        x, y, w, h = region
        processed = gray.copy()

        if 'rotation' in problems:
            processed = self._fix_rotation(processed, region)
            self.add_step("Fixed Rotation", processed)
            new_region = self._detect_barcode_region(processed)
            if new_region is None:
                print("Barcode region lost after rotation.")
                return processed
            x, y, w, h = new_region

        if 'foreign_object' in problems:
            processed = self._remove_foreign_object(processed, (x,y,w,h))
            self.add_step("Removed Foreign Object", processed)

        if 'frequency_domain' in problems:
            processed = self._fix_frequency_domain(processed, (x,y,w,h))
            self.add_step("Fixed Frequency Domain", processed)

        if 'compression_artifacts' in problems:
            processed = self._fix_compression_artifacts(processed, (x,y,w,h))
            self.add_step("Fixed Compression Artifacts", processed)

        if 'salt_pepper' in problems:
            processed = self._remove_salt_pepper(processed, (x,y,w,h))
            self.add_step("Removed Salt & Pepper", processed)

        if 'blur' in problems:
            processed = self._fix_blur(processed, (x,y,w,h))
            self.add_step("Fixed Blur", processed)

        if 'low_contrast' in problems:
            processed = self._fix_contrast(processed, (x,y,w,h))
            self.add_step("Enhanced Contrast", processed)

        roi = processed[y:y+h, x:x+w]
        binary = self._binarize(roi)
        binary = self._validate_barcode_elements(binary)
        self.add_step("Binarized", binary)

        cleaned = self._final_barcode_cleanup(binary)
        self.add_step("Final Cleanup", cleaned)

        processed[y:y+h, x:x+w] = cleaned
        symbology = self._determine_barcode_symbology(cleaned)
        print(f"Detected barcode symbology: {symbology}")

        return processed

    def visualize_steps(self):
        """Visualize pipeline steps in a grid."""
        n_steps=len(self.steps)
        cols=min(4, n_steps)
        rows=(n_steps+cols-1)//cols

        fig, axes=plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows*cols==1:
            axes=np.array([axes])
        if rows==1:
            axes=axes.reshape(1,-1)
        elif cols==1:
            axes=axes.reshape(-1,1)

        for i,(desc,img) in enumerate(self.steps):
            row=i//cols
            col=i%cols
            ax=axes[row, col]
            if len(img.shape)==2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(desc, fontsize=8)
            ax.axis('off')

        for i in range(len(self.steps), rows*cols):
            row=i//cols
            col=i%cols
            axes[row,col].axis('off')

        plt.tight_layout()
        plt.show()

# -------------------------------------------------------------------
# Multi-Pipeline Manager
# -------------------------------------------------------------------

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
        """Process image through all pipelines, evaluate, and pick best."""
        self.original_image = image.copy()
        self.all_results = []

        # Run each pipeline
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

        # Choose best
        self.best_result = max(self.all_results, key=lambda x: x.quality_score)
        return self.best_result.processed_image

    def visualize_all_results(self):
        """5-row figure:
        row=0 => original image
        row=1 => pipeline1 steps
        row=2 => pipeline2 steps
        row=3 => pipeline3 steps
        row=4 => winner pipeline final result + metrics."""
        if not self.all_results or self.original_image is None:
            print("No results to visualize.")
            return

        # Among pipelines, find max # steps:
        max_steps = max(len(r.steps) for r in self.all_results)
        rows=5
        cols=max_steps

        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

        # Row 0 => original
        axes[0,0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        # turn off the rest
        for c in range(1, cols):
            axes[0,c].axis('off')

        # Row=1..3 => pipeline steps
        for row_index, pipeline_result in enumerate(self.all_results, start=1):
            for col_index in range(cols):
                ax=axes[row_index, col_index]
                if col_index<len(pipeline_result.steps):
                    desc,img = pipeline_result.steps[col_index]
                    if len(img.shape)==2:
                        ax.imshow(img, cmap='gray')
                    else:
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"{pipeline_result.name}: {desc}", fontsize=8)
                ax.axis('off')

        # row=4 => winner (col=0 => final image, col=1 => metrics)
        winner_img = self.best_result.processed_image
        axes[4,0].imshow(winner_img, cmap='gray')
        axes[4,0].set_title(f"Winner: {self.best_result.name}\nScore: {self.best_result.quality_score:.2f}", fontsize=8)
        axes[4,0].axis('off')

        # Show metrics in col=1
        metrics_text = "\n".join([
            f"Edge Score: {self.best_result.metrics['edge_score']:.2f}",
            f"Contrast Score: {self.best_result.metrics['contrast_score']:.2f}",
            f"Pattern Score: {self.best_result.metrics['pattern_score']:.2f}",
            f"Noise Score: {self.best_result.metrics['noise_score']:.2f}"
        ])
        axes[4,1].text(0.05, 0.5, metrics_text, fontsize=10, va='center')
        axes[4,1].axis('off')

        for c in range(2, cols):
            axes[4,c].axis('off')

        plt.tight_layout()
        plt.savefig("pipeline_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def print_statistics(self):
        """Print stats for all pipelines."""
        print("\n=== Pipeline Processing Statistics ===")
        for result in self.all_results:
            print(f"\n{result.name}:")
            print(f"Overall Quality Score: {result.quality_score:.2f}")
            print("Detailed Metrics:")
            for metric_name, metric_value in result.metrics.items():
                print(f"  {metric_name}: {metric_value:.2f}")

        print(f"\nWinning Pipeline: {self.best_result.name}")
        print(f"Winning Score: {self.best_result.quality_score:.2f}")

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
            processor.print_statistics()
            print("\nVisualization saved as 'pipeline_comparison.png'")
        else:
            print("Failed to load image.")
    else:
        print("No image selected.")