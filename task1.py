import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk


# Hossam's Task: Perspective Correction
def perspective_correction_pipeline(image):
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
    return image  # Return the original image if no valid perspective correction is found.


# Kassab's Task: Preprocessing
def preprocessing_pipeline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    blurred = cv2.medianBlur(gray, 3)
    binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)[1]
    return binary 


# Monica's Task: Cropping
def cropping_pipeline(binary, original_image):
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
        cropped = original_image[y_start:y_end, x_start:x_end]
        return cropped
    return original_image


# # Patrick's Task: Final Processing
# def final_processing_pipeline(cropped_barcode):
#     gray_cropped = cv2.cvtColor(cropped_barcode, cv2.COLOR_BGR2GRAY)
#     binary_cropped = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     cleaned = cv2.morphologyEx(binary_cropped, cv2.MORPH_CLOSE, kernel)
#     return cleaned


# Load Image
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


def hide_tkinter_window():
    tk.Tk().withdraw()


# Main Function 
def process_and_display(image, integrated=False):
    # Initialize placeholders for each step
    perspective_corrected = None
    binary = None
    cropped_barcode = None
    cleaned_barcode = None

    # Step 1: Perspective Correction
    try:
        perspective_corrected = perspective_correction_pipeline(image)
    except Exception as e:
        print(f"Error in Perspective Correction: {e}")
        perspective_corrected = image  # Fallback to the original image

    # Step 2: Preprocessing
    if perspective_corrected is not None:
        try:
            binary = preprocessing_pipeline(perspective_corrected)
        except Exception as e:
            print(f"Error in Preprocessing: {e}")

    # Step 3: Cropping
    if binary is not None:
        try:
            cropped_barcode = cropping_pipeline(binary, binary)
        except Exception as e:
            print(f"Error in Cropping: {e}")
            cropped_barcode = perspective_corrected  # Fallback to perspective-corrected image

    # Step 4: Final Processing
    if cropped_barcode is not None:
        try:
            cleaned_barcode = final_processing_pipeline(cropped_barcode)
        except Exception as e:
            print(f"Error in Final Processing: {e}")

    # Integrated Output (Final Result Only)
    if integrated:
        if cleaned_barcode is not None:
            plt.figure(figsize=(8, 6))
            plt.title("Final Processed Barcode")
            plt.imshow(cleaned_barcode, cmap="gray")
            plt.axis("off")
            plt.show()
        else:
            print("Pipeline incomplete: Final result cannot be displayed.")
        return

    # Display all steps for testing individual components
    steps = []
    titles = []

    # Append each step only if it is completed
    steps.append(image)
    titles.append("Original Image")

    if perspective_corrected is not None:
        steps.append(perspective_corrected)
        titles.append("Perspective Corrected")

    if binary is not None:
        steps.append(binary)
        titles.append("Binary (Otsu)")

    if cropped_barcode is not None:
        steps.append(cropped_barcode)
        titles.append("Cropped Barcode")

    if cleaned_barcode is not None:
        steps.append(cleaned_barcode)
        titles.append("Final Processed Barcode")

    # Plot the results
    fig, axes = plt.subplots(1, len(steps), figsize=(20, 6))
    for i, (step, title) in enumerate(zip(steps, titles)):
        ax = axes[i]
        ax.imshow(step if len(step.shape) == 2 else cv2.cvtColor(step, cv2.COLOR_BGR2RGB), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image = load_image()
    if image is None:
        exit()

    # Set integrated=True to display only the final result, or False for testing individual components.
    process_and_display(image, integrated=False)
