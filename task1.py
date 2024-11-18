import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Initialize Tkinter for image selection
tk.Tk().withdraw()

# Step 0: Select an image
input_image_path = filedialog.askopenfilename(title="Select a Barcode Image",
                                              filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

if not input_image_path:
    print("No file selected. Exiting...")
    exit()

# Function to save each step as an image
def save_step(image, step_name):
    cv2.imwrite(f"{step_name.replace(' ', '_').lower()}.png", image)

# Function to display all steps in a summary grid
def display_summary(images, titles):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Function to check background contrast
def background_contrast_test(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist /= hist.sum()
    # Calculate the ratio of bright pixels
    white_ratio = hist[200:].sum()  # Bright pixels (values above 200)
    
    # Heuristic threshold to determine if background is uniform or noisy
    return white_ratio > 0.4

# Step 1: Load the image
image = cv2.imread(input_image_path)
original_image = image.copy()

# Step 2: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Detect Edges
edges = cv2.Canny(gray, 50, 150)
save_step(edges, "Edges Detected")

# Step 4: Find Contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

apply_perspective = background_contrast_test(image)

if apply_perspective:
    print("White background detected: Applying perspective correction.")
    # Step 5: Find the Largest Quadrilateral for Perspective Correction
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:  # Quadrilateral check
        pts = approx.reshape(4, 2)

        # Sort points to consistent order: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        # Compute dimensions
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Perspective transformation
        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original_image, matrix, (max_width, max_height))
        save_step(warped, "Warped Image")

        # Proceed with noise removal and cropping
        processed_image = warped
    else:
        print("No valid quadrilateral found. Skipping perspective correction.")
        processed_image = image
else:
    print("Low contrast detected: Skipping perspective correction.")
    processed_image = image

# Convert to Grayscale
gray_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

# Apply Median Blur
blurred = cv2.medianBlur(gray_processed, 5)
save_step(blurred, "Blurred Image")

# Apply Thresholding
_, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
save_step(binary, "Binary Image")

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
save_step(cleaned, "Cleaned Binary Image")

# Display Summary
steps = [original_image, edges, processed_image, gray_processed, blurred, binary, cleaned]
titles = ["Original Image", "Edges Detected", "Processed Image", "Grayscale", "Blurred", "Binary", "Cleaned Binary"]
display_summary(steps, titles)
