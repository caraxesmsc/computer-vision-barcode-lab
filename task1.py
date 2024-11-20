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

def process_and_display(image):
    # Original image
    original_image = image.copy()

    # Step 1: Correct perspective
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            rect = np.zeros((4, 2), dtype="float32")
            pts = approx.reshape(4, 2)
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left

            # Calculate width and height
            width_a = np.linalg.norm(rect[2] - rect[3])
            width_b = np.linalg.norm(rect[1] - rect[0])
            max_width = max(int(width_a), int(width_b))
            height_a = np.linalg.norm(rect[1] - rect[2])
            height_b = np.linalg.norm(rect[0] - rect[3])
            max_height = max(int(height_a), int(height_b))

            # Perform perspective transform
            dst = np.array(
                [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                dtype="float32",
            )
            matrix = cv2.getPerspectiveTransform(rect, dst)
            perspective_corrected = cv2.warpPerspective(image, matrix, (max_width, max_height))
        else:
            perspective_corrected = image
    else:
        perspective_corrected = image

    # Step 2: Preprocessing
    gray = cv2.cvtColor(perspective_corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Step 3: Crop barcode area
    vertical_projection = np.sum(binary == 0, axis=0)
    horizontal_projection = np.sum(binary == 0, axis=1)

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
        x_end = min(x_end + padding_x, binary.shape[1])
        y_start = max(0, y_start - padding_y)
        y_end = min(y_end + padding_y, binary.shape[0])
        cropped_barcode = perspective_corrected[y_start:y_end, x_start:x_end]
    else:
        cropped_barcode = perspective_corrected

    # Step 4: Final processing of the cropped barcode
    gray_cropped = cv2.cvtColor(cropped_barcode, cv2.COLOR_BGR2GRAY)
    binary_cropped = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary_cropped, cv2.MORPH_CLOSE, kernel)

    # Display all steps together
    steps = [
        original_image,
        perspective_corrected,
        blurred,
        binary,
        cropped_barcode,
        binary_cropped,
        cleaned,
    ]
    titles = [
        "Original Image",
        "Perspective Corrected",
        "median 5X5",
        "Binary (Otsu)",
        "Cropped Barcode",
        "Binary Cropped",
        "Morphological Cleaning",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (image, title) in enumerate(zip(steps, titles)):
        ax = axes[i // 4, i % 4]
        ax.imshow(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    cv2.imwrite("cleaned_barcode.png", cleaned)


if __name__ == "__main__":
    image = load_image()
    if image is None:
        exit()
    process_and_display(image)
