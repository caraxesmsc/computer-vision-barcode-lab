import os
import cv2 as cv
import numpy as np
import matplotlib
from scipy.signal import find_peaks

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tkinter import filedialog
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MatLike = Union[np.ndarray, None]

@dataclass
class ExtractionResult:
    file_name: str
    original: np.ndarray
    barcode: Optional[np.ndarray]

##########################################
# Image Processing Functions
##########################################

def turnGrayscale(image: np.ndarray) -> np.ndarray:
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return img_gray

def detectSineWaveNoise(image: np.ndarray) -> bool:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    avg_freq = np.mean(magnitude_spectrum)

    # Filter out low magnitude frequencies
    low_freq_threshold = avg_freq * 1.5
    magnitude_spectrum[magnitude_spectrum < low_freq_threshold] = 0

    peak_threshold = np.max(magnitude_spectrum) * 0.1
    peaks = np.where(magnitude_spectrum > peak_threshold)

    # if not enough peaks => no sine wave noise
    if len(peaks[0]) < 2:
        return False

    peak_distances = []
    for i in range(1, len(peaks[0])):
        distance = np.linalg.norm(
            np.array([peaks[0][i], peaks[1][i]])
            - np.array([peaks[0][i - 1], peaks[1][i - 1]])
        )
        peak_distances.append(distance)

    mean_distance = np.mean(peak_distances)
    # If the distance between peaks is fairly consistent => likely a sine wave
    regularity = (np.std(peak_distances) < mean_distance * 0.2)

    return True if regularity else False

def adaptiveThresholding(
        image: np.ndarray,
        block_size: int = 15,
        C: int = 2,
        method: str = 'gaussian'
) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if method.lower() == 'gaussian':
        adaptive_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        adaptive_method = cv.ADAPTIVE_THRESH_MEAN_C

    thresholded_image = cv.adaptiveThreshold(
        image,
        255,
        adaptive_method,
        cv.THRESH_BINARY,
        block_size,
        C
    )
    return thresholded_image

def rotateBarcode(image: np.ndarray) -> np.ndarray:
    blurred_image = cv.GaussianBlur(image, (5, 5), 0)
    edges = cv.Canny(blurred_image, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)

        rect = cv.minAreaRect(largest_contour)
        box = cv.boxPoints(rect)
        box = np.int32(box)

        angle = rect[-1]
        if angle < -45:
            angle += 90

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])

        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        img_rotated = cv.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        if new_w < new_h:
            img_rotated_90 = cv.rotate(img_rotated, cv.ROTATE_90_CLOCKWISE)
        else:
            img_rotated_90 = img_rotated
        return img_rotated_90
    else:
        return image

def medianFilter(image: np.ndarray) -> np.ndarray:
    img_blur = cv.blur(image, (1, 17))
    img_filtered = cv.medianBlur(img_blur, 5)

    img_filtered = np.uint8(np.clip(img_filtered, 0, 255))

    _, img_thresh = cv.threshold(img_filtered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return img_thresh

def histogramStretching(image: np.ndarray) -> np.ndarray:
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image.copy()

    stretched = np.uint8((image - min_val) / (max_val - min_val) * 255)
    return stretched

def extractBarcode(image: np.ndarray, aggressive_crop: bool = False) -> np.ndarray:
    img_thresh_inv = cv.bitwise_not(image)
    x, y, w, h = cv.boundingRect(img_thresh_inv)

    if aggressive_crop:
        cut_amount_vertical = h // 8  # More aggressive vertical cut
        cut_amount_horizontal = w // 8  # More aggressive horizontal cut
        barcode = image[y + cut_amount_vertical: y + h - cut_amount_vertical,
                  x + cut_amount_horizontal: x + w - cut_amount_horizontal]
    else:
        cut_amount = min(h // 4, h)
        barcode = image[y: y + h - cut_amount, x: x + w]
    return barcode

def closeOpen(image: np.ndarray) -> np.ndarray:
    kernel_height = image.shape[0] // 5
    kernel_width = 3
    kernel = np.ones((kernel_height, kernel_width), np.uint8)

    img_closed = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=1)
    img_opened = cv.morphologyEx(img_closed, cv.MORPH_OPEN, kernel, iterations=1)
    return img_opened

def decode_code11_barcode(image: np.ndarray, tolerance: float) -> list:
    NARROW = "0"
    WIDE = "1"
    code11_widths = {
        "00110": "Stop/Start",
        "10001": "1",
        "01001": "2",
        "11000": "3",
        "00101": "4",
        "10100": "5",
        "01100": "6",
        "00011": "7",
        "10010": "8",
        "10000": "9",
        "00001": "0",
        "00100": "-"
    }

    mean_col = image.mean(axis=0)
    mean_col[mean_col <= 127] = 1
    mean_col[mean_col > 128] = 0

    pixels = ''.join(mean_col.astype(np.uint8).astype(str))

    narrow_bar_size = 0
    for px in pixels:
        if px == "1":
            narrow_bar_size += 1
        else:
            break

    wide_bar_size = narrow_bar_size * 2
    min_narrow = narrow_bar_size * (1 - tolerance)
    max_narrow = narrow_bar_size * (1 + tolerance)
    min_wide = wide_bar_size * (1 - tolerance)
    max_wide = wide_bar_size * (1 + tolerance)

    digits = []
    pixel_index = 0
    current_digit_widths = ""
    skip_next = False

    while pixel_index < len(pixels):
        if skip_next:
            pixel_index += int(narrow_bar_size)
            skip_next = False
            continue

        count = 1
        try:
            while pixels[pixel_index] == pixels[pixel_index + 1]:
                count += 1
                pixel_index += 1
        except IndexError:
            pass
        pixel_index += 1

        if min_narrow <= count <= max_narrow:
            current_digit_widths += NARROW
        elif min_wide <= count <= max_wide:
            current_digit_widths += WIDE
        else:
            # If count doesn't match narrow or wide, reset the pattern
            current_digit_widths = ""

        if current_digit_widths in code11_widths:
            digits.append(code11_widths[current_digit_widths])
            current_digit_widths = ""
            skip_next = True

    return digits

def detect_foreign_object(image: np.ndarray) -> bool:
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    vertical_profile = np.mean(image, axis=0)
    v_std = np.std(vertical_profile)
    v_mean = np.mean(vertical_profile)
    return np.any(np.abs(vertical_profile - v_mean) > 2.5 * v_std)

#########################
# Cropping Pipelines
#########################
def cropping_pipeline_bounding_rect(binary: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Crops the image based on the bounding rectangle of the largest contour.
    """
    inv = cv.bitwise_not(binary)
    contours, _ = cv.findContours(inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original_image

    # Find the largest contour and its bounding rectangle
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)

    cropped = original_image[y:y+h, x:x+w]
    return cropped

def aggressive_crop(image: np.ndarray, v_thresh_factor: float = 0.05, h_thresh_factor: float = 0.05, padding: int = 2) -> np.ndarray:
    """
    Aggressive cropping using projections with variable thresholds and padding.
    """
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    v_proj = np.sum(binary == 0, axis=0)
    h_proj = np.sum(binary == 0, axis=1)

    v_thresh = np.mean(v_proj) * v_thresh_factor
    h_thresh = np.mean(h_proj) * h_thresh_factor

    x_start_candidates = np.where(v_proj > v_thresh)[0]
    x_end_candidates = np.where(v_proj > v_thresh)[0]
    y_start_candidates = np.where(h_proj > h_thresh)[0]
    y_end_candidates = np.where(h_proj > h_thresh)[0]

    x_start = x_start_candidates[0] if x_start_candidates.size > 0 else 0
    x_end = x_end_candidates[-1] if x_end_candidates.size > 0 else image.shape[1] - 1
    y_start = y_start_candidates[0] if y_start_candidates.size > 0 else 0
    y_end = y_end_candidates[-1] if y_end_candidates.size > 0 else image.shape[0] - 1

    x_start = max(0, x_start - padding)
    x_end = min(image.shape[1], x_end + padding)
    y_start = max(0, y_start - padding)
    y_end = min(image.shape[0], y_end + padding)

    return gray[y_start:y_end, x_start:x_end]

#########################
# Foreign Object Removal
#########################
def remove_foreign_object(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # 1) Remove foreign horizontal objects
    thresh = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
    hp = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(hp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # e.g., remove wide but short horizontal lines
        if w > 50 and h < 20:
            cv.rectangle(gray_image, (x, y), (x + w, y + h), (255), -1)

    # 2) Remove foreign vertical objects
    thresh = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 25))
    vp = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=3)
    cnts = cv.findContours(vp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # e.g., remove tall but very narrow vertical lines
        if h > 50 and w < 20:
            cv.rectangle(gray_image, (x, y), (x + w, y + h), (255), -1)

    # 3) Mild blur to smooth edges
    result = cv.medianBlur(gray_image, 5)

    # 4) Initial boundingRect crop
    _, binary_result = cv.threshold(result, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    cropped_result = cropping_pipeline_bounding_rect(binary_result, result)

    # 5) Extra crop step (using aggressive_crop)
    final_cropped = aggressive_crop(cropped_result)

    return final_cropped

# --------------------------------------------------------------------------
# Parameters for tuning (kept at the top level for easy access)
# --------------------------------------------------------------------------
AGGRESSIVE_CROP_V_THRESH_FACTOR = 0.05  # Vertical threshold factor for aggressive cropping
AGGRESSIVE_CROP_H_THRESH_FACTOR = 0.05  # Horizontal threshold factor for aggressive cropping
AGGRESSIVE_CROP_PADDING = 2            # Padding for aggressive cropping

# --------------------------------------------------------------------------

def preprocess_image(image: np.ndarray) -> np.ndarray:
    img_gray = turnGrayscale(image)

    sine_wave_noise_detected = detectSineWaveNoise(img_gray)
    if sine_wave_noise_detected:
        img_gray = adaptiveThresholding(img_gray)

    img_rotated = rotateBarcode(img_gray)
    img_thresh = medianFilter(img_rotated)
    img_stretched = histogramStretching(img_thresh)
    img_extracted = extractBarcode(img_stretched)
    img_closed_opened = closeOpen(img_extracted)

    return img_closed_opened

def extract_barcode_with_retry(image: np.ndarray, file_name: str, expected_barcode: List[str]) -> Tuple[
    np.ndarray, List[str]]:
    # 1. Standard Preprocessing + Decode
    preprocessed = preprocess_image(image)
    decoded = decode_code11_barcode(preprocessed, 0.3)
    decoded_numbers = [digit for digit in decoded if digit not in ["Stop/Start"]]
    expected_numbers = [digit for digit in expected_barcode if digit not in ["Stop/Start"]]

    if decoded_numbers == expected_numbers:
        return preprocessed, decoded

    # 2. Detect Foreign Object -> remove -> decode
    if detect_foreign_object(preprocessed):
        logger.info(f"Foreign object detected in {file_name}. Removing and retrying.")
        removed_fo = remove_foreign_object(preprocessed)
        decoded_fo = decode_code11_barcode(removed_fo, 0.3)
        decoded_fo_numbers = [digit for digit in decoded_fo if digit not in ["Stop/Start"]]
        if decoded_fo_numbers == expected_numbers:
            return removed_fo, decoded_fo

    # 3. Aggressive Cropping + Decode
    logger.info(f"Standard decoding failed for {file_name}. Trying aggressive crop.")
    aggressively_cropped = aggressive_crop(
        preprocessed,
        v_thresh_factor=AGGRESSIVE_CROP_V_THRESH_FACTOR,
        h_thresh_factor=AGGRESSIVE_CROP_H_THRESH_FACTOR,
        padding=AGGRESSIVE_CROP_PADDING
    )
    decoded_aggressive = decode_code11_barcode(aggressively_cropped, 0.3)
    decoded_aggressive_numbers = [d for d in decoded_aggressive if d not in ["Stop/Start"]]
    if decoded_aggressive_numbers == expected_numbers:
        return aggressively_cropped, decoded_aggressive

    # 4. Aggressive Crop + FO Removal + Decode
    logger.info(f"Aggressive crop failed for {file_name}. Trying aggressive crop with foreign object removal.")
    aggressively_cropped_fo = remove_foreign_object(aggressively_cropped)
    decoded_aggressive_fo = decode_code11_barcode(aggressively_cropped_fo, 0.3)
    decoded_aggressive_fo_numbers = [d for d in decoded_aggressive_fo if d not in ["Stop/Start"]]
    if decoded_aggressive_fo_numbers == expected_numbers:
        return aggressively_cropped_fo, decoded_aggressive_fo

    # 5. If all else fails, return the result from FO removal
    return aggressively_cropped_fo, decoded_aggressive_fo_numbers

def extract_barcode_for_plotting(image: np.ndarray, file_name: str) -> Optional[np.ndarray]:
    cleaned = preprocess_image(image)
    return cleaned

def plot_image(file_name: str,
               original_image: np.ndarray,
               barcode_image: Optional[np.ndarray] = None) -> None:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {file_name}")
    plt.axis("off")
    if len(original_image.shape) == 2:
        plt.imshow(original_image, cmap='gray')
    else:
        plt.imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title(f"Extracted Barcode: {file_name}")
    plt.axis("off")
    if barcode_image is not None:
        plt.imshow(barcode_image, cmap='gray')
    else:
        print(f"No barcode detected for {file_name}.")

    plt.savefig("debug_" + file_name.replace(" ", "_") + ".png", dpi=100)
    plt.close()

def extract_barcodes(folder_path: str):
    images = os.listdir(folder_path)
    for file_name in images:
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        image_path = os.path.join(folder_path, file_name)
        image = cv.imread(image_path)
        if image is not None:
            try:
                barcode_image = extract_barcode_for_plotting(image, file_name)
                plot_image(file_name, image, barcode_image)
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
        else:
            logger.warning(f"Could not read image: {file_name}")

if __name__ == "__main__":
    print("Please select a folder with test images.")
    folder_selected = filedialog.askdirectory(title="Select Folder with Test Images")
    if not folder_selected:
        print("[INFO] No folder selected. Exiting.")
        exit()

    # 2) Example barcodes (expected results)
    expected_barcodes = [
        ['Stop/Start', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', 'Stop/Start'],
        ['Stop/Start', '1', '0', '4', '-', '1', '1', '6', '-', '1', '1', '6', 'Stop/Start'],
        ['Stop/Start', '1', '1', '2', '-', '1', '1', '5', '-', '5', '8', '-', 'Stop/Start'],
        ['Stop/Start', '-', '4', '7', '-', '4', '7', '-', '1', '2', '1', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '1', '-', '1', '1', '7', '-', '1', '1', '6', 'Stop/Start'],
        ['Stop/Start', '-', '1', '1', '7', '-', '4', '6', '-', '9', '8', '-', 'Stop/Start'],
        ['Stop/Start', '1', '0', '1', '-', '4', '7', '-', '1', '0', '0', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '3', '-', '1', '1', '9', '-', '5', '2', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '9', '-', '5', '7', '-', '1', '1', '9', '-', 'Stop/Start'],
        ['Stop/Start', '1', '0', '3', '-', '1', '2', '0', '-', '9', '9', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '3', '-', '4', '7', '-', '3', '5', '-', '3', '5', 'Stop/Start']
    ]

    num_pass = 0
    num_fail = 0

    images = os.listdir(folder_selected)
    for i, file_name in enumerate(images):
        if i >= len(expected_barcodes):
            break

        image_path = os.path.join(folder_selected, file_name)
        image = cv.imread(image_path)

        barcode_image, decoded = extract_barcode_with_retry(image, file_name, expected_barcodes[i])
        expected = expected_barcodes[i]
        test_outcome = "Pass" if (decoded == expected) else "Fail"

        logger.info(f"Decoded barcode for image {i + 1}: {decoded}")
        logger.info(f"Test {i + 1} ({file_name}) => {test_outcome}")

        if test_outcome == "Pass":
            num_pass += 1
        else:
            num_fail += 1

        plot_image(file_name, image, barcode_image)

    logger.info(f"Summary: {num_pass} tests passed, {num_fail} tests failed")

    extract_barcodes(folder_selected)
    print("Done.")