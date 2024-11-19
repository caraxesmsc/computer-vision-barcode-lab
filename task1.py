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
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axes[i // 4, i % 4]
        ax.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped

def refine_binary(binary_image):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    return opened

def crop_barcode_area(preprocessed_image, original_image):
    vertical_projection = np.sum(preprocessed_image == 0, axis=0)
    horizontal_projection = np.sum(preprocessed_image == 0, axis=1)

    vertical_threshold = 0.6 * np.max(vertical_projection)
    horizontal_threshold = 0.6 * np.max(horizontal_projection)

    vertical_indices = np.where(vertical_projection > vertical_threshold)[0]
    horizontal_indices = np.where(horizontal_projection > horizontal_threshold)[0]

    if len(vertical_indices) == 0 or len(horizontal_indices) == 0:
        print("Barcode region not detected.")
        return original_image

    x_start, x_end = vertical_indices[0], vertical_indices[-1]
    y_start, y_end = horizontal_indices[0], horizontal_indices[-1]

    padding_x = int(0.05 * (x_end - x_start))
    padding_y = int(0.05 * (y_end - y_start))

    x_start = max(0, x_start - padding_x)
    x_end = min(x_end + padding_x, preprocessed_image.shape[1])
    y_start = max(0, y_start - padding_y)
    y_end = min(y_end + padding_y, preprocessed_image.shape[0])

    cropped = original_image[y_start:y_end, x_start:x_end]
    return cropped

def process_barcode(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(cleaned, kernel, iterations=1)

    plt.figure(figsize=(15, 10))
    steps = [cropped_image, blurred, binary, cleaned, dilated]
    titles = ["Original Image", "Blurred", "Binary Thresholding", "Morphological Cleaning", "Dilated Output"]

    for i, (step, title) in enumerate(zip(steps, titles)):
        plt.subplot(2, 3, i + 1)
        plt.title(title)
        plt.imshow(step if len(step.shape) == 2 else cv2.cvtColor(step, cv2.COLOR_BGR2RGB), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = load_image()
    if image is None:
        exit()

    original_image = image.copy()
    perspective_corrected = correct_perspective(image)

    gray = cv2.cvtColor(perspective_corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    refined_binary = refine_binary(binary)
    cropped_barcode = crop_barcode_area(refined_binary, perspective_corrected)

    steps = [
        original_image, perspective_corrected, gray, blurred, binary, refined_binary, cropped_barcode
    ]
    titles = [
        "Original Image", "Perspective Corrected", "Grayscale", "Blurred", "Binary", "Refined Binary", "Cropped Barcode"
    ]
    display_summary(steps, titles)

    process_barcode(cropped_barcode)
