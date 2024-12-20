import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        """Save each step for visualization"""
        self.steps.append((description, image.copy()))

    def estimate_bar_thickness(self, image):
        """Estimate barcode bar thickness after mild preprocessing"""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        mid_slice = sobelx[sobelx.shape[0] // 4:3 * sobelx.shape[0] // 4]
        for row in mid_slice:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3  # Default fallback
        min_width = max(1, int(np.percentile(widths, 10)))
        return min_width

    def detect_barcode_region(self, image):
        """Find the barcode region using vertical edges"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        blur = cv2.GaussianBlur(gradx, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            aspect = w / h
            roi = gradx[y:y + h, x:x + w]
            edge_density = np.sum(roi > 50) / (w * h)
            if 1.5 < aspect < 5 and edge_density > 0.1:
                valid_regions.append((x, y, w, h))

        if not valid_regions:
            return None
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]))

    def detect_image_problems(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return [], (0, 0, gray.shape[1], gray.shape[0])

        x, y, w, h = barcode_region
        roi = gray[y:y + h, x:x + w]

        # Blur detection
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.mean(edges > 0)
        blur = (laplacian_var < 30 and edge_density < 0.15)

        # Salt & Pepper
        median = cv2.medianBlur(roi, 3)
        noise_diff = np.abs(roi - median)
        salt_pepper = (np.percentile(noise_diff, 95) > 70)

        # Frequency domain
        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        freq_problem = np.std(magnitude_spectrum) > 70

        # Low contrast
        p5, p95 = np.percentile(roi, [5, 95])
        low_contrast = (p95 - p5) < 50

        # Rotation detection
        coords = np.column_stack(np.where(edges > 0))
        rotation = False
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 5:
                rotation = True

        # Foreign object detection heuristic
        min_bar_width = self.estimate_bar_thickness(roi)
        vertical_profile = np.mean(roi, axis=0)
        window_size = min_bar_width if min_bar_width>0 else 3
        if len(vertical_profile) > window_size:
            local_stds = [vertical_profile[i:i + window_size].std() for i in range(len(vertical_profile) - window_size)]
            v_std = np.std(local_stds)
        else:
            v_std = 0
        foreign_object = v_std > 15

        problems = []
        if foreign_object:
            problems.append('foreign_object')
        if rotation:
            problems.append('rotation')
        if freq_problem:
            problems.append('frequency_domain')
        if salt_pepper:
            problems.append('salt_pepper')
        if blur:
            problems.append('blur')
        if low_contrast:
            problems.append('low_contrast')

        return problems, barcode_region

    def fix_rotation(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y + h, x:x + w]
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            rows, cols = roi.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated = cv2.warpAffine(roi, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            result = image.copy()
            result[y:y + h, x:x + w] = rotated
            return result
        return image

    def fix_frequency_domain(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y + h, x:x + w].copy()

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = roi.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        xx, yy = np.ogrid[:rows, :cols]
        mask_area = (xx - crow) ** 2 + (yy - ccol) ** 2 <= r * r
        mask[mask_area] = 0
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        cleaned = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result = image.copy()
        result[y:y + h, x:x + w] = cleaned
        return result

    def remove_salt_pepper(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y + h, x:x + w].copy()
        filtered = cv2.medianBlur(roi, 3)
        filtered = cv2.medianBlur(filtered, 5)
        result = image.copy()
        result[y:y + h, x:x + w] = filtered
        return result

    def fix_blur(self, image, barcode_region):
        x, y, w, h = barcode_region
        roi = image[y:y+h, x:x+w].copy()
        gaussian = cv2.GaussianBlur(roi,(0,0),3)
        unsharp = cv2.addWeighted(roi,1.3,gaussian,-0.3,0)
        result = image.copy()
        result[y:y+h, x:x+w]=unsharp
        return result

    def fix_contrast(self, image, barcode_region):
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        enhanced=clahe.apply(roi)
        result=image.copy()
        result[y:y+h,x:x+w]=enhanced
        return result

    def binarize(self, image):
        p2,p98=np.percentile(image,(2,98))
        normalized=np.interp(image,(p2,p98),(0,255)).astype(np.uint8)
        binary=cv2.adaptiveThreshold(
            normalized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,51,5
        )
        kernel=np.ones((2,2),np.uint8)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        return binary

    def remove_foreign_object_after_binarization(self, binary):
        black_ratios=np.mean((binary==0).astype(float),axis=0)
        median_ratio=np.median(black_ratios)
        threshold=np.std(black_ratios)*2
        anomalous=np.abs(black_ratios - median_ratio)>threshold
        bar_width=self.estimate_bar_thickness(binary)
        bar_width = max(1, bar_width)
        # Just replace anomalous cols with nearest good col
        cleaned=binary.copy()
        h,w=binary.shape
        for i in range(w):
            if anomalous[i]:
                # find nearest good cols
                left_good=None
                for j in range(i-1,-1,-1):
                    if not anomalous[j]:
                        left_good=j
                        break
                right_good=None
                for j in range(i+1,w):
                    if not anomalous[j]:
                        right_good=j
                        break
                
                if left_good is not None and right_good is not None:
                    replacement=(cleaned[:,left_good].astype(float)+cleaned[:,right_good].astype(float))/2
                    cleaned[:,i]=replacement
                elif left_good is not None:
                    cleaned[:,i]=cleaned[:,left_good]
                elif right_good is not None:
                    cleaned[:,i]=cleaned[:,right_good]
        return cleaned

    def final_crop(self,binary):
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary
        largest=max(contours,key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(largest)
        pad_x=int(0.05*w)
        pad_y=int(0.1*h)
        x=max(0,x-pad_x)
        y=max(0,y-pad_y)
        w=min(binary.shape[1]-x,w+2*pad_x)
        h=min(binary.shape[0]-y,h+2*pad_y)
        return binary[y:y+h,x:x+w]

    def preprocess_image(self, image):
        self.steps=[]
        self.add_step("Original Image",image)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image.copy()

        problems, barcode_region=self.detect_image_problems(image)
        self.add_step("Detected Problems: "+", ".join(problems), gray)

        if barcode_region is None:
            # fallback: just binarize entire image
            binary=self.binarize(gray)
            self.add_step("Binarized",binary)
            final=self.final_crop(binary)
            self.add_step("Final Crop",final)
            return final

        processed=gray.copy()
        x,y,w,h=barcode_region

        # Rotation first
        if 'rotation' in problems:
            processed=self.fix_rotation(processed,barcode_region)
            self.add_step("Fixed Rotation",processed)
            new_reg=self.detect_barcode_region(processed)
            if new_reg:
                x,y,w,h=new_reg

        # Frequency domain
        if 'frequency_domain' in problems:
            processed=self.fix_frequency_domain(processed,(x,y,w,h))
            self.add_step("Fixed Frequency Domain",processed)

        # Salt & pepper
        if 'salt_pepper' in problems:
            processed=self.remove_salt_pepper(processed,(x,y,w,h))
            self.add_step("Removed Salt & Pepper",processed)

        # Blur
        if 'blur' in problems:
            processed=self.fix_blur(processed,(x,y,w,h))
            self.add_step("Fixed Blur",processed)

        # Low contrast
        if 'low_contrast' in problems:
            processed=self.fix_contrast(processed,(x,y,w,h))
            self.add_step("Enhanced Contrast",processed)

        # Binarize ROI now
        roi=processed[y:y+h,x:x+w]
        binary=self.binarize(roi)
        self.add_step("Binarized ROI",binary)

        # foreign object after binarization
        if 'foreign_object' in problems:
            binary=self.remove_foreign_object_after_binarization(binary)
            self.add_step("Removed Foreign Object (Post-Binarization)",binary)

        final=self.final_crop(binary)
        self.add_step("Final Crop",final)
        return final

    def visualize_steps(self):
        n_steps=len(self.steps)
        cols=min(4,n_steps)
        rows=(n_steps+cols-1)//cols
        fig, axes=plt.subplots(rows,cols,figsize=(4*cols,4*rows))
        if rows*cols==1:
            axes=[axes]
        else:
            if rows==1:
                axes=axes.reshape(1,-1)
            elif cols==1:
                axes=axes.reshape(-1,1)
        for i,(desc,img) in enumerate(self.steps):
            r=i//cols
            c=i%cols
            ax=axes[r,c]
            if len(img.shape)==2:
                ax.imshow(img,cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            ax.set_title(desc,fontsize=8)
            ax.axis('off')
        for i in range(len(self.steps),rows*cols):
            r=i//cols
            c=i%cols
            axes[r,c].axis('off')
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    preprocessor=BarcodePreprocessor()
    image_path=filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files","*.png;*.jpg;*.jpeg")]
    )
    if image_path:
        image=cv2.imread(image_path)
        if image is not None:
            processed=preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()
