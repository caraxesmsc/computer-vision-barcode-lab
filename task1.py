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
        plt.subplot(3, 3, i + 1)  # Change 2, 4 to 3, 3 for a 3x3 grid
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

# Load the image
image = cv2.imread(input_image_path)

if image is None:
    print("Image not found. Check the file path.")
    exit()

original_image = image.copy()

# Step 1: Apply perspective correction
processed_image, edges, debug_image = perspective_correction_with_debug(image)

# Step 2: Convert to grayscale and continue processing
# Post-processing after perspective correction with improved noise removal
def post_process_barcode(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Apply Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    
    # Adaptive Thresholding for dynamic binarization
    adaptive_binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(adaptive_binary, cv2.MORPH_CLOSE, kernel)
    
    # Additional erosion and dilation for further cleanup
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_binary = cv2.dilate(morphed, kernel2, iterations=1)
    cleaned_binary = cv2.erode(cleaned_binary, kernel2, iterations=1)
    
    return gray, enhanced_gray, blurred, adaptive_binary, cleaned_binary


# Process the corrected image
gray, enhanced_gray, blurred, adaptive_binary, cleaned_binary = post_process_barcode(processed_image)

# Display all steps including the improved post-processing
steps = [original_image, edges, debug_image, processed_image, gray, enhanced_gray, blurred, adaptive_binary, cleaned_binary]
titles = ["Original Image", "Edges Detected", "Debug Contour", "Processed Image",
          "Grayscale", "Enhanced Contrast", "Blurred", "Adaptive Threshold", "Cleaned Binary"]
display_summary(steps, titles)
