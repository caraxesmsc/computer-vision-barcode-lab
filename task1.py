import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

# Initialize Tkinter for image selection
tk.Tk().withdraw()

# Step 0: Select an image
input_image_path = filedialog.askopenfilename(title="Select a Barcode Image",
                                              filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

if not input_image_path:
    print("No file selected. Exiting...")
    exit()

# Function to display all steps in a summary grid
def display_summary(images, titles):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Perspective correction with debug
def perspective_correction_with_debug(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Adaptive thresholding to enhance contrast between paper and background
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours detected.")
        return image, edges, image.copy()  # Return original image as debug_image

    # Filter contours based on area and aspect ratio
    paper_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Ignore small contours
            continue

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if it’s a quadrilateral
        if len(approx) == 4:
            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:  # Ensure it's roughly rectangular
                paper_contour = approx
                break

    debug_image = image.copy()  # Initialize debug image
    if paper_contour is None:
        print("No valid paper contour found.")
        return image, edges, debug_image  # Return image without perspective correction

    # Draw detected contour on debug image
    cv2.drawContours(debug_image, [paper_contour], -1, (0, 255, 0), 2)

    # Order points in a consistent manner (top-left, top-right, bottom-right, bottom-left)
    pts = paper_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    # Compute dimensions of the new perspective
    width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped, edges, debug_image

# Refinement process
def refine_binary_image(binary_image):
    # Morphological opening to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)
    
    # Skeletonization to thin the barcode lines
    skeleton = cv2.ximgproc.thinning(opened, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # Contour filtering to remove small, irrelevant noise
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_binary = np.zeros_like(skeleton)
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Threshold for minimum contour area
            cv2.drawContours(filtered_binary, [contour], -1, 255, thickness=cv2.FILLED)
    
    return opened, skeleton, filtered_binary

# Load the image
image = cv2.imread(input_image_path)

if image is None:
    print("Image not found. Check the file path.")
    exit()

original_image = image.copy()

# Step 1: Apply perspective correction
processed_image, edges, debug_image = perspective_correction_with_debug(image)

# Step 2: Convert to grayscale and continue processing
gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply CLAHE and adaptive thresholding
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
adaptive_binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Step 4: Refine the binary image
opened, skeleton, filtered_binary = refine_binary_image(adaptive_binary)

# Display all steps in a grid
steps = [
    original_image, edges, debug_image, processed_image, gray,
    enhanced_gray, blurred, adaptive_binary, opened, skeleton, filtered_binary
]
titles = [
    "Original Image", "Edges Detected", "Debug Contour", "Processed Image", "Grayscale",
    "Enhanced Contrast", "Blurred", "Adaptive Threshold", "Opened", "Skeletonized", "Filtered Binary"
]
display_summary(steps, titles)
