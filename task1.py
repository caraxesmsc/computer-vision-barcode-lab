import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk


def hide_tkinter_window():
    tk.Tk().withdraw()


def load_image():
    hide_tkinter_window()
    image_path = filedialog.askopenfilename(
        title="Select a Barcode Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if not image_path:
        print("No file selected. Exiting...")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found. Check the file path.")
        return None
    return image


def display_summary(images, titles):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axes[i // 5, i % 5]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours detected.")
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) != 4:
        print("No valid quadrilateral found.")
        return image

    rect = np.zeros((4, 2), dtype="float32")
    pts = approx.reshape(4, 2)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    width_a = np.linalg.norm(rect[2] - rect[3])
    width_b = np.linalg.norm(rect[1] - rect[0])
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(rect[1] - rect[2])
    height_b = np.linalg.norm(rect[0] - rect[3])
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def refine_threshold(binary_image):
    """Apply morphological operations to clean up the thresholded image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return cleaned


def crop_barcode_area(thresholded_image, original_image):
    """Crop the barcode area based on vertical line density."""
    vertical_projection = np.sum(thresholded_image == 0, axis=0)
    threshold = 0.6 * np.max(vertical_projection)
    barcode_indices = np.where(vertical_projection > threshold)[0]
    if len(barcode_indices) == 0:
        print("Barcode region not detected.")
        return original_image

    start, end = barcode_indices[0], barcode_indices[-1]
    height, _ = thresholded_image.shape

    #added 10 pixels to the start and end to make sure the barcode is fully captured
    start = max(0, start - 10)
    end = min(end + 10, height)

    cropped = original_image[:, start:end]
    return cropped


if __name__ == "__main__":
    image = load_image()
    if image is None:
        exit()

    original_image = image.copy()

    # Step 1: Correct perspective
    perspective_corrected = correct_perspective(image)

    # Step 2: Preprocess
    gray = cv2.cvtColor(perspective_corrected, cv2.COLOR_BGR2GRAY)
    median_3 = cv2.medianBlur(gray, 3)
    median_5 = cv2.medianBlur(median_3, 5)
    median_3_2 = cv2.medianBlur(median_5, 3)
    median_5_2 = cv2.medianBlur(median_3_2, 5)
    Gausian3 = cv2.GaussianBlur(median_5_2, (3, 3), 0)


    adaptive_binary = cv2.adaptiveThreshold(
        Gausian3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    refined_binary = refine_threshold(adaptive_binary)

    # Step 3: Crop barcode area
    cropped_barcode = crop_barcode_area(refined_binary, perspective_corrected)

    # Display results
    steps = [original_image, perspective_corrected, median_3, median_5, median_3_2, median_5_2,  Gausian3, adaptive_binary, refined_binary, cropped_barcode]
    titles = [
        "Original Image",
        "Perspective Correction",
        "Median 3x3",
        "Median 5x5",
        "Median 3x3",
        "Median 5x5",
        "Gaussian 3x3",
        "Adaptive Binary",
        "Refined Binary",
        "Cropped Barcode",
    ]
    display_summary(steps, titles)
