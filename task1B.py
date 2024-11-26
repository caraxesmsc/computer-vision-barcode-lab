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

def to_grayscale(image):
    if len(image.shape) == 3:  # Check if multi-channel
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Improved Detection Functions
def detect_foreign_object(image, threshold=0.3):
    gray = to_grayscale(image)
    vertical_projection = gray.mean(axis=0)
    horizontal_projection = gray.mean(axis=1)
    
    # Detect significant dips in projection that might indicate obstruction
    vertical_anomalies = np.where(vertical_projection < threshold * np.median(vertical_projection))[0]
    horizontal_anomalies = np.where(horizontal_projection < threshold * np.median(horizontal_projection))[0]
    
    return len(vertical_anomalies) > 0 or len(horizontal_anomalies) > 0

def detect_blur(image, threshold=50):
    gray = to_grayscale(image)
    # Use Variance of Laplacian and additional checks
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    fm = laplacian_var / (1.0 + np.std(gray))
    return fm < threshold

def detect_tilt(image, angle_threshold=10):
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is None:
        return False
    
    # Collect line angles
    line_angles = []
    for line in lines:
        for rho, theta in line:
            angle = np.degrees(theta)
            line_angles.append(abs(90 - angle))
    
    # Check if any angle significantly deviates from vertical
    return any(abs(angle) > angle_threshold for angle in line_angles)

def correct_foreign_object(image):
    gray = to_grayscale(image)
    
    # Advanced inpainting method
    mask = np.zeros(gray.shape, dtype=np.uint8)
    vertical_projection = gray.mean(axis=0)
    horizontal_projection = gray.mean(axis=1)
    
    # Create mask for low-intensity regions
    vertical_mask = vertical_projection < 0.3 * np.median(vertical_projection)
    horizontal_mask = horizontal_projection < 0.3 * np.median(horizontal_projection)
    
    mask[:, np.where(vertical_mask)[0]] = 255
    mask[np.where(horizontal_mask)[0], :] = 255
    
    # Use advanced inpainting
    corrected = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return corrected

def correct_blur(image, method='sharp'):
    if method == 'sharp':
        # Advanced unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return unsharp
    elif method == 'deconv':
        # Wiener deconvolution (more advanced but computationally intensive)
        kernel = np.array([[1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9]])
        image_float = image.astype(np.float32) / 255.0
        deconvolved = cv2.filter2D(image_float, -1, kernel)
        return (deconvolved * 255).astype(np.uint8)

def correct_tilt(image, tolerate_threshold=5):
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is None:
        return image
    
    for line in lines:
        for rho, theta in line:
            angle = np.degrees(theta)
            
            # Check if tilt is significant
            if abs(90 - angle) > tolerate_threshold:
                center = (image.shape[1] // 2, image.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, 90 - angle, 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, 
                                         (image.shape[1], image.shape[0]), 
                                         flags=cv2.INTER_LINEAR)
                return rotated
    
    return image

def detect_and_correct(image, verbose=True):
    steps = [("Original Image", image)]
    logs = []

    # Foreign Object Correction
    if detect_foreign_object(image):
        logs.append("Removing obstructions...")
        image = correct_foreign_object(image)
        steps.append(("Obstruction Removed", image))

    # Blur Correction
    if detect_blur(image):
        logs.append("Enhancing image sharpness...")
        image = correct_blur(image, method='sharp')
        steps.append(("Image Sharpened", image))

    # Tilt Correction (more conservative)
    if detect_tilt(image):
        logs.append("Correcting image orientation...")
        image = correct_tilt(image)
        steps.append(("Orientation Corrected", image))

    # Contrast Enhancement
    image = cv2.equalizeHist(to_grayscale(image))
    steps.append(("Contrast Enhanced", image))

    if verbose:
        visualize_steps(steps, logs)

    return image

# Visualization remains the same as in the original script
def visualize_steps(steps, logs):
    fig, axes = plt.subplots(1, len(steps), figsize=(20, 6))
    for i, (title, step_image) in enumerate(steps):
        ax = axes[i]
        ax.imshow(
            step_image if len(step_image.shape) == 2 else cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB),
            cmap="gray"
        )
        ax.set_title(title)
        ax.axis("off")
    plt.show()
    print("\n".join(logs))

# Main Function
if __name__ == "__main__":
    image = load_image()
    if image is None:
        exit()

    print("Detecting and correcting issues...")
    corrected_image = detect_and_correct(image)