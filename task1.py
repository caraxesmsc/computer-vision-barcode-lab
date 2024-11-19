import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

# Helper function to hide the Tkinter window
def hide_tkinter_window():
    tk.Tk().withdraw()

# Load an image selected by the user
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

# Display all processing steps in a grid
def display_summary(images, titles):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Correct perspective distortion with debugging information
def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours detected.")
        return image, cv2.Canny(blurred, 50, 150), image.copy()

    paper_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:
                paper_contour = approx
                break

    debug_image = image.copy()
    if paper_contour is None:
        print("No valid contour found.")
        return image, cv2.Canny(blurred, 50, 150), debug_image

    cv2.drawContours(debug_image, [paper_contour], -1, (0, 255, 0), 2)
    pts = paper_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped, cv2.Canny(blurred, 50, 150), debug_image

# Crop the barcode area using connected components
def crop_barcode_area(binary_image, original_image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    max_area = 0
    barcode_bbox = None
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h
        if 1.5 < aspect_ratio < 8.0 and area > 500:
            if area > max_area:
                max_area = area
                barcode_bbox = (x, y, w, h)
    if barcode_bbox is None:
        print("Barcode area not detected properly.")
        return binary_image
    x, y, w, h = barcode_bbox
    cropped = original_image[y : y + h, x : x + w]
    return cropped

# Main execution
if __name__ == "__main__":
    image = load_image()
    if image is None:
        exit()

    original_image = image.copy()
    processed_image, edges, debug_image = correct_perspective(image)

    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    median_filtered = cv2.medianBlur(gray, 5)
    adaptive_binary = cv2.adaptiveThreshold(
        median_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(adaptive_binary, cv2.MORPH_OPEN, kernel)

    cropped_barcode = crop_barcode_area(opened, processed_image)

    steps = [
        original_image,
        cv2.Canny(gray, 50, 150),
        debug_image,
        processed_image,
        median_filtered,
        adaptive_binary,
        opened,
        cropped_barcode,
    ]
    titles = [
        "Original Image",
        "Edges",
        "Debug Contour",
        "Processed Image",
        "Median Filtered",
        "Adaptive Threshold",
        "Opened",
        "Cropped Barcode",
    ]
    display_summary(steps, titles)
