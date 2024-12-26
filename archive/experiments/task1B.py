import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def hide_tkinter_window():
    tk.Tk().withdraw()


def load_image():
    hide_tkinter_window()
    image_path = filedialog.askopenfilename(
        title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if not image_path:
        print("No file selected. Exiting...")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found. Check the file path.")
        return None
    return image, image_path


def to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


### Detection Functions ###
def detect_foreign_object(image, threshold=0.1):
    gray = to_grayscale(image)
    vertical_projection = gray.mean(axis=0)
    dips = np.where(vertical_projection < threshold * np.median(vertical_projection))[0]
    return len(dips) > 0


def detect_blur(image, threshold=100):
    gray = to_grayscale(image)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def detect_low_contrast(image, low_threshold=50, high_threshold=200):
    gray = to_grayscale(image)
    min_val, max_val = np.min(gray), np.max(gray)
    return (max_val - min_val) < (high_threshold - low_threshold)


def detect_salt_and_pepper(image, threshold=0.02):
    gray = to_grayscale(image)
    total_pixels = gray.size
    unique, counts = np.unique(gray, return_counts=True)
    noise_pixels = sum(counts[np.where((unique == 0) | (unique == 255))])
    noise_ratio = noise_pixels / total_pixels
    return noise_ratio > threshold


def detect_frequency_noise_refined(image, threshold=0.1):
    gray = to_grayscale(image)
    dft = np.fft.fft2(gray)
    fshift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(1 + np.abs(fshift))
    median_magnitude = np.median(magnitude_spectrum)
    mask = magnitude_spectrum > (median_magnitude * (1 + threshold))
    noise_ratio = np.sum(mask) / gray.size
    return noise_ratio > 0.05, mask


### Correction Functions ###
def correct_foreign_object(image):
    gray = to_grayscale(image)
    vertical_avg = gray.mean(axis=0)
    mask = gray < (0.1 * vertical_avg.mean())
    inpainted = cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return inpainted


def correct_blur(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return unsharp


def enhance_contrast(image, method="adaptive"):
    gray = to_grayscale(image)
    if method == "adaptive":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    elif method == "stretch":
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def correct_salt_and_pepper(image):
    return cv2.medianBlur(image, 5)


def correct_frequency_noise(image, mask=None):
    gray = to_grayscale(image)
    dft = np.fft.fft2(gray)
    fshift = np.fft.fftshift(dft)
    if mask is None:
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back).astype(np.uint8)


### Main Pipeline ###
def detect_and_correct_refined(image):
    steps = [("Original Image", image)]
    logs = []

    if detect_foreign_object(image):
        logs.append("Fixing foreign object...")
        image = correct_foreign_object(image)
        steps.append(("Foreign Object Fixed", image))

    if detect_blur(image):
        logs.append("Sharpening image...")
        image = correct_blur(image)
        steps.append(("Blur Fixed", image))

    if detect_low_contrast(image):
        logs.append("Enhancing contrast...")
        image = enhance_contrast(image)
        steps.append(("Contrast Enhanced", image))

    if detect_salt_and_pepper(image):
        logs.append("Removing salt-and-pepper noise...")
        image = correct_salt_and_pepper(image)
        steps.append(("Noise Removed", image))

    is_noisy, noise_mask = detect_frequency_noise_refined(image)
    if is_noisy:
        logs.append("Correcting frequency noise...")
        image = correct_frequency_noise(image, noise_mask)
        steps.append(("Frequency Noise Fixed", image))

    return image, steps, logs


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


### Execution ###
if __name__ == "__main__":
    image, path = load_image()
    if image is not None:
        corrected_image, steps, logs = detect_and_correct_refined(image)
        visualize_steps(steps, logs)
