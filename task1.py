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
        # Adjust color mapping based on image type
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def correct_perspective(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edged = cv2.Canny(blurred, 50, 200)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours detected.")
        return image

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Look for a quadrilateral
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # Map the quadrilateral to a rectangle
            rect = np.zeros((4, 2), dtype="float32")
            pts = approx.reshape(4, 2)
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left

            # Compute the dimensions of the new image
            width_a = np.linalg.norm(rect[2] - rect[3])
            width_b = np.linalg.norm(rect[1] - rect[0])
            max_width = max(int(width_a), int(width_b))
            height_a = np.linalg.norm(rect[1] - rect[2])
            height_b = np.linalg.norm(rect[0] - rect[3])
            max_height = max(int(height_a), int(height_b))

            # Destination points for perspective transform
            dst = np.array(
                [
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1],
                ],
                dtype="float32",
            )
            # Apply perspective transformation
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
            return warped
    print("No valid quadrilateral found.")
    return image


def refine_threshold(binary_image):
    """Apply morphological operations to clean up the thresholded image."""
    # Larger kernel for closing to fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Smaller kernel for opening to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Close small holes inside the foreground objects
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    return opened


def crop_barcode_area(thresholded_image, original_image):
    """Crop the barcode area based on connected components."""
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("No contours found.")
        return original_image

    # Get the largest contour which should be the barcode
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Add padding to the bounding box
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(w + 2 * padding, original_image.shape[1] - x)
    h = min(h + 2 * padding, original_image.shape[0] - y)
    # Crop the barcode area from the original image
    cropped = original_image[y : y + h, x : x + w]
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
    # Apply median blur to reduce noise while preserving edges
    median_blurred = cv2.medianBlur(gray, 5)
    # Further smooth the image with Gaussian blur
    gaussian_blurred = cv2.GaussianBlur(median_blurred, (5, 5), 0)

    # Step 3: Thresholding using Otsu's method
    _, binary = cv2.threshold(
        gaussian_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert the image if the barcode is lighter than the background
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Step 4: Refine the thresholded image
    refined_binary = refine_threshold(binary)

    # Step 5: Crop barcode area
    cropped_barcode = crop_barcode_area(refined_binary, perspective_corrected)

    # Display results
    steps = [
        original_image,
        perspective_corrected,
        gray,
        median_blurred,
        gaussian_blurred,
        binary,
        refined_binary,
        cropped_barcode,
    ]
    titles = [
        "Original Image",
        "Perspective Correction",
        "Grayscale",
        "Median Blurred",
        "Gaussian Blurred",
        "Otsu's Thresholding",
        "Refined Binary",
        "Cropped Barcode",
    ]
    display_summary(steps, titles)
