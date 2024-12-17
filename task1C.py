import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os
import json
from datetime import datetime
import logging

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.debug = False
        self.log_data = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BarcodePreprocessor')

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def process_folder(self, folder_path):
        """Process all images in a folder and generate log"""
        self.logger.info(f"Starting batch processing in folder: {folder_path}")
        
        # Initialize batch log
        batch_log = {
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "folder_path": folder_path,
            "processed_images": []
        }
        
        # Process each image
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(folder_path, filename)
                self.logger.info(f"Processing image: {filename}")
                
                # Process single image and collect results
                image_log = self.process_single_image(image_path)
                batch_log["processed_images"].append(image_log)
                
                # Save processing steps visualization
                self.save_visualization(os.path.join(folder_path, f"processed_{filename}"))
        
        # Save detailed log file
        log_filename = os.path.join(folder_path, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_filename, 'w') as f:
            json.dump(batch_log, f, indent=4)
        
        # Generate summary
        self.generate_summary_report(batch_log, folder_path)
        
        self.logger.info(f"Batch processing completed. Log saved to: {log_filename}")
        return batch_log

    def process_single_image(self, image_path):
        """Process a single image and return its log data"""
        image_log = {
            "filename": os.path.basename(image_path),
            "path": image_path,
            "problems_detected": [],
            "filters_applied": [],
            "processing_success": False,
            "error": None
        }
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Reset steps for new image
            self.steps = []
            
            # Pre-handle overexposure (if needed)
            # If image is very bright, try gamma correction with gamma < 1
            # This is a heuristic step:
            if np.mean(image) > 200:  # Example threshold for overexposure
                image = self.handle_overexposure(image)
                self.add_step("Handled Overexposure", image)
            
            # Process image
            processed_image = self.preprocess_image(image)
            
            # Update log with processing details
            # Detected problems step is the second step in self.steps (index 1)
            # If no problems were detected, the string would be "Detected Problems: "
            detected_str = self.steps[1][0].replace("Detected Problems: ", "").strip()
            if detected_str:
                image_log["problems_detected"] = detected_str.split(", ")
            else:
                image_log["problems_detected"] = []
            
            # Filters applied: steps after problem detection
            filters_applied = [s[0] for s in self.steps[2:] if not s[0].startswith("Detected Problems")]
            image_log["filters_applied"] = filters_applied
            image_log["processing_success"] = True
            
            # Save processed image
            output_path = os.path.join(os.path.dirname(image_path), f"processed_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, processed_image)
            image_log["output_path"] = output_path
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            image_log["error"] = str(e)
            image_log["processing_success"] = False
        
        return image_log

    def generate_summary_report(self, batch_log, folder_path):
        """Generate a human-readable summary report"""
        total_images = len(batch_log["processed_images"])
        successful_processes = sum(1 for img in batch_log["processed_images"] if img["processing_success"])
        
        # Collect statistics
        all_problems = []
        all_filters = []
        for img in batch_log["processed_images"]:
            if img["processing_success"]:
                all_problems.extend(img["problems_detected"])
                all_filters.extend(img["filters_applied"])
        
        problem_stats = {p: all_problems.count(p) for p in set(all_problems) if p}
        filter_stats = {f: all_filters.count(f) for f in set(all_filters) if f}
        
        # Generate report
        report = f"""BARCODE PREPROCESSING SUMMARY REPORT
Generated: {batch_log['processing_date']}
Folder: {batch_log['folder_path']}

PROCESSING STATISTICS
Total Images Processed: {total_images}
Successful: {successful_processes} ({(successful_processes/total_images)*100:.1f}%)
Failed: {total_images - successful_processes}

PROBLEMS DETECTED
{'-'*50}"""
        
        for problem, count in problem_stats.items():
            report += f"\n{problem}: {count} images ({(count/total_images)*100:.1f}%)"
        
        report += f"\n\nFILTERS APPLIED\n{'-'*50}"
        for filter_name, count in filter_stats.items():
            report += f"\n{filter_name}: {count} images ({(count/total_images)*100:.1f}%)"
        
        # Save report
        report_path = os.path.join(folder_path, f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Summary report generated: {report_path}")

    def detect_image_problems(self, image):
        """Improved problem detection with better thresholds and analysis"""
        problems = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Improved blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:
            problems.append('blur')

        # Enhanced contrast detection
        p5, p95 = np.percentile(gray, [5, 95])
        if (p95 - p5) < 50:
            problems.append('low_contrast')

        # Improved salt and pepper detection
        median = cv2.medianBlur(gray, 3)
        diff = np.abs(gray - median)
        if np.percentile(diff, 95) > 30:
            problems.append('salt_pepper')

        # Frequency domain problem detection (basic approach)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift) + 1)
        if np.std(magnitude_spectrum) < 20:
            problems.append('frequency_domain')

        # Improved rotation detection
        rotation_angle = self.get_rotation_angle(gray)
        if abs(rotation_angle) > 5:
            problems.append('rotation')

        # Foreign object detection only after other issues are checked
        if not any(p in problems for p in ['blur', 'frequency_domain']):
            if self.detect_missing_sections(gray):
                problems.append('foreign_object')

        return problems

    def detect_missing_sections(self, gray):
        """Detect missing or covered sections (foreign objects)"""
        vertical_profile = np.mean(gray, axis=0)
        smoothed_profile = np.convolve(vertical_profile, np.ones(5)/5, mode='same')
        
        threshold = np.mean(smoothed_profile) * 0.45

        gaps = np.where(smoothed_profile < threshold)[0]
        if len(gaps) > 0:
            gap_groups = np.split(gaps, np.where(np.diff(gaps) > 10)[0] + 1)
            significant_gaps = [g for g in gap_groups if len(g) > gray.shape[1]*0.05]
            if significant_gaps:
                return True
        return False

    def get_rotation_angle(self, gray):
        """Improved rotation angle detection using Hough transform with preprocessing"""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        kernel = np.ones((3,3),np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations = 1)
        
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            return np.median(angles) if angles else 0
        return 0

    def fix_frequency_domain(self, gray):
        """Fix frequency domain issues with a refined high-pass filter.
        
        NOTE: For more complex frequency domain issues, consider using notch filters or 
        directional filters. This is a placeholder for advanced techniques.
        """
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Gaussian high-pass filter
        x = np.linspace(-ccol, ccol-1, cols)
        y = np.linspace(-crow, crow-1, rows)
        xv, yv = np.meshgrid(x, y)
        sigma = 30
        gaussian_filter = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
        mask = 1 - gaussian_filter

        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def remove_foreign_object(self, gray):
        """Improved foreign object removal using barcode line interpolation with weighted approach"""
        vertical_profile = np.mean(gray, axis=0)

        # Smooth the profile for peak/valley detection
        window_size = 5
        smoothed_profile = np.convolve(vertical_profile, np.ones(window_size)/window_size, mode='same')

        # Adjust parameters for find_peaks
        peaks, _ = find_peaks(smoothed_profile, prominence=5, distance=3)
        valleys, _ = find_peaks(-smoothed_profile, prominence=5, distance=3)

        # Combine and sort
        extrema = np.sort(np.concatenate((peaks, valleys)))

        # If no extrema found, return original
        if len(extrema) == 0:
            return gray

        threshold = np.mean(vertical_profile) * 0.45
        problem_regions = np.where(vertical_profile < threshold)[0]

        cleaned = np.copy(gray)

        if len(problem_regions) > 0:
            gap_groups = np.split(problem_regions, np.where(np.diff(problem_regions) > 1)[0] + 1)

            for group in gap_groups:
                if len(group) > gray.shape[1] * 0.05:  # Significant gap
                    start = group[0]
                    end = group[-1]

                    # Find nearest valid lines/spaces
                    left_extrema = extrema[extrema < start]
                    right_extrema = extrema[extrema > end]

                    if len(left_extrema) > 0 and len(right_extrema) > 0:
                        for i in range(start, end):
                            left_idx = np.searchsorted(extrema, i, side="right") - 1
                            right_idx = np.searchsorted(extrema, i, side="left")

                            if left_idx >= 0 and right_idx < len(extrema):
                                dist_left = i - extrema[left_idx]
                                dist_right = extrema[right_idx] - i

                                weight_left = 1 / dist_left if dist_left > 0 else 1
                                weight_right = 1 / dist_right if dist_right > 0 else 1

                                left_val = np.mean(gray[:, max(0, extrema[left_idx] - 2):extrema[left_idx]+3])
                                right_val = np.mean(gray[:, extrema[right_idx]-2:min(gray.shape[1], extrema[right_idx] + 3)])

                                cleaned[:, i] = (left_val * weight_left + right_val * weight_right) / (weight_left + weight_right)
                            elif left_idx >= 0:
                                # Only left extrema
                                cleaned[:, i] = np.mean(gray[:, max(0, extrema[left_idx] - 2):extrema[left_idx]+3])
                            elif right_idx < len(extrema):
                                # Only right extrema
                                cleaned[:, i] = np.mean(gray[:, extrema[right_idx]-2:min(gray.shape[1], extrema[right_idx] + 3)])
        return cleaned

    def remove_salt_and_pepper(self, gray):
        """Adaptive median filtering for salt and pepper noise removal"""
        threshold = 30  # Adjust threshold as needed
        padded_gray = np.pad(gray, 1, mode='edge')
        cleaned = np.copy(gray)

        rows, cols = gray.shape
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                neighborhood = padded_gray[i-1:i+2, j-1:j+2].flatten()
                center_pixel = padded_gray[i, j]
                # Remove center pixel before median
                neighborhood_wo_center = np.delete(neighborhood, 4)
                median_val = np.median(neighborhood_wo_center)

                # If pixel significantly differs from median, replace it
                if abs(center_pixel - median_val) > threshold:
                    cleaned[i-1, j-1] = median_val

        return cleaned

    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline with improved foreign object removal and adaptive filtering"""
        self.steps = []
        self.add_step("Original Image", image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Detect problems
        problems = self.detect_image_problems(image)
        self.add_step("Detected Problems: " + ", ".join(problems), gray)
        
        # Remove foreign objects if detected
        if 'foreign_object' in problems and self.detect_missing_sections(gray):
            gray = self.remove_foreign_object(gray)
            self.add_step("Removed Foreign Object", gray)
        
        # Fix frequency domain issues first if any
        if 'frequency_domain' in problems:
            gray = self.fix_frequency_domain(gray)
            self.add_step("Fixed Frequency Domain", gray)
        
        # Fix blur - more aggressive approach plus sharpening
        if 'blur' in problems:
            # Attempt a mild sharpening after a Gaussian blur
            gray_blurred = cv2.GaussianBlur(gray, (0, 0), 3)
            gray = cv2.addWeighted(gray, 1.5, gray_blurred, -0.5, 0)
            # Further sharpening with an unsharp mask
            # Kernel for unsharp mask
            kernel = np.array([[ -1, -1, -1],
                               [ -1,  9, -1],
                               [ -1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel)
            self.add_step("Fixed Blur (Sharpened)", gray)
        
        # Fix rotation if significant
        if 'rotation' in problems:
            angle = self.get_rotation_angle(gray)
            if abs(angle) > 5:
                gray = self.fix_rotation(gray, angle)
                self.add_step("Fixed Rotation", gray)
                
        # Fix contrast issues
        if 'low_contrast' in problems:
            gray = self.fix_contrast(gray)
            self.add_step("Enhanced Contrast", gray)
        
        # Fix salt and pepper noise last with adaptive median filtering
        if 'salt_pepper' in problems:
            gray = self.remove_salt_and_pepper(gray)
            self.add_step("Removed Salt and Pepper", gray)
        
        # Final binary conversion with adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        self.add_step("Binary Image", binary)
        
        # Morphological cleanup
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.add_step("Cleaned Binary", binary)
        
        # Crop to barcode
        cropped = self.crop_to_barcode(binary)
        self.add_step("Final Result", cropped)
        
        return cropped

    @staticmethod
    def fix_contrast(image):
        """Improved contrast enhancement using adaptive CLAHE and gamma correction"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Gamma correction
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        return enhanced
        
    def fix_rotation(self, image, angle):
        """Improved rotation correction"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Use linear interpolation for smoothness
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        return rotated
    
    @staticmethod
    def crop_to_barcode(binary):
        """Improved cropping with better component analysis"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return binary
        
        valid_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0
            area = cv2.contourArea(cnt)
            if 2 < aspect_ratio < 4 and area > (binary.shape[0] * binary.shape[1] * 0.1):
                valid_contours.append((cnt, area))
        
        if not valid_contours:
            return binary
        
        # Use the largest valid contour
        largest_cnt = max(valid_contours, key=lambda x: x[1])[0]
        x, y, w, h = cv2.boundingRect(largest_cnt)
        
        padding = int(min(w, h) * 0.05)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary.shape[1] - x, w + 2 * padding)
        h = min(binary.shape[0] - y, h + 2 * padding)
        
        return binary[y:y+h, x:x+w]

    def save_visualization(self, output_path):
        """Save visualization of processing steps"""
        plt.figure(figsize=(20, 10))
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        for i, (description, image) in enumerate(self.steps):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                      cmap='gray')
            plt.title(description, fontsize=8, wrap=True)
            plt.axis('off')
        
        plt.tight_layout()
        viz_path = f"{os.path.splitext(output_path)[0]}_steps.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_steps(self):
        """Visualize processing steps in interactive mode"""
        n_steps = len(self.steps)
        cols = min(5, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        plt.figure(figsize=(20, 10))
        for i, (description, image) in enumerate(self.steps):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                      cmap='gray')
            plt.title(description, fontsize=8, wrap=True)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def handle_overexposure(self, image):
        """Handle overexposed images with gamma correction < 1 and optional clipping."""
        # Optional clipping
        clipped = np.clip(image, 0, 240)
        
        # Gamma correction with gamma < 1 to darken
        gamma = 0.7
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(clipped, table)
        return corrected

if __name__ == "__main__":
    # Initialize Tk root window and hide it
    root = Tk()
    root.withdraw()
    
    # Create preprocessor instance
    preprocessor = BarcodePreprocessor()
    
    # Ask for folder selection
    folder_path = filedialog.askdirectory(
        title="Select Folder Containing Barcode Images"
    )
    
    if folder_path:
        try:
            # Process all images in folder
            batch_log = preprocessor.process_folder(folder_path)
            print(f"Processing completed. Check {folder_path} for results and logs.")
        except Exception as e:
            print(f"An error occurred during processing: {str(e)}")
    else:
        print("No folder selected. Exiting.")
