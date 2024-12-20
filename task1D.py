import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []

    def add_step(self, description, image):
        self.steps.append((description, image.copy()))

    def aggressive_stretch(self, img):
        min_val, max_val = np.min(img), np.max(img)
        if max_val == min_val:
            return img
        return ((img - min_val)/(max_val - min_val)*255).astype(np.uint8)

    def estimate_bar_thickness(self, image):
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        widths = []
        mid_slice = sobelx[sobelx.shape[0]//4:3*sobelx.shape[0]//4]
        for row in mid_slice:
            peaks, _ = find_peaks(row, height=np.mean(row))
            if len(peaks) > 1:
                widths.extend(np.diff(peaks))

        if not widths:
            return 3, 9
        min_width = max(1, int(np.percentile(widths, 10)))
        pattern_width = max(3, int(np.median(widths)))
        return min_width, pattern_width

    def detect_barcode_region(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        blur = cv2.GaussianBlur(gradx, (21,21),0)
        _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid=[]
        for cnt in contours:
            x,y,w,h=cv2.boundingRect(cnt)
            if w<10 or h<10:
                continue
            aspect=w/h
            roi=gradx[y:y+h,x:x+w]
            edge_density=np.sum(roi>50)/(w*h)
            if 1.5<aspect<5 and edge_density>0.1:
                valid.append((x,y,w,h))
        if not valid:
            return None
        return max(valid,key=lambda r:np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def detect_image_problems(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return [], (0,0,gray.shape[1],gray.shape[0])

        x,y,w,h=barcode_region
        roi=gray[y:y+h,x:x+w]

        min_bar_width, pattern_width = self.estimate_bar_thickness(roi)

        # Foreign object detection (heuristic)
        vertical_profile=np.mean(roi,axis=0)
        # If variance of local std is high:
        window_size=pattern_width
        if len(vertical_profile)>window_size:
            local_stds=[vertical_profile[i:i+window_size].std() for i in range(len(vertical_profile)-window_size)]
            v_std = np.std(local_stds)
        else:
            v_std=0
        foreign_object = v_std>15

        # Blur detection
        lap_var=cv2.Laplacian(roi,cv2.CV_64F).var()
        edges=cv2.Canny(roi,50,150)
        edge_density=np.mean(edges>0)
        blur=(lap_var<50 and edge_density<0.1)

        # Low contrast
        p5,p95=np.percentile(roi,[5,95])
        low_contrast=(p95 - p5)<60

        # Salt & Pepper detection
        median=cv2.medianBlur(roi,3)
        noise_diff=np.abs(roi-median)
        # Stricter threshold
        salt_pepper=np.percentile(noise_diff,95)>70

        # Frequency domain detection
        f_transform=np.fft.fft2(roi)
        f_shift=np.fft.fftshift(f_transform)
        magnitude_spectrum=20*np.log(np.abs(f_shift)+1)
        freq_problem=np.std(magnitude_spectrum)>70

        # Rotation detection
        lines=cv2.HoughLines(edges,1,np.pi/180,threshold=min(w,h)//3)
        rotation=False
        if lines is not None:
            angles=[]
            for rho,theta in lines[:,0]:
                angle=np.degrees(theta)
                if angle<45:
                    angles.append(angle)
                elif angle>135:
                    angles.append(angle-180)
            if angles and abs(np.median(angles))>5:
                rotation=True

        problems=[]
        if foreign_object:
            problems.append('foreign_object')
        if rotation:
            problems.append('rotation')
        if freq_problem:
            problems.append('frequency_domain')
        # Skipping compression detection to reduce false positives
        if salt_pepper:
            problems.append('salt_pepper')
        if blur:
            problems.append('blur')
        if low_contrast:
            problems.append('low_contrast')

        return problems, barcode_region

    def fix_rotation(self, image, barcode_region):
        # Codebase 1 approach: just rotate based on minAreaRect
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w]
        edges=cv2.Canny(roi,50,150)
        coords=np.column_stack(np.where(edges>0))
        if len(coords)>0:
            angle=cv2.minAreaRect(coords)[-1]
            if angle<-45:
                angle=90+angle
            rows, cols=roi.shape
            M=cv2.getRotationMatrix2D((cols/2,rows/2), angle,1)
            rotated=cv2.warpAffine(roi,M,(cols,rows),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
            result=image.copy()
            result[y:y+h,x:x+w]=rotated
            return result
        return image

    def fix_frequency_domain(self, image, barcode_region):
        # Gaussian high-pass filter (like codebase 3)
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        f_transform=np.fft.fft2(roi)
        f_shift=np.fft.fftshift(f_transform)
        rows,cols=roi.shape
        crow,ccol=rows//2,cols//2
        sigma=30
        xv,yv=np.meshgrid(np.linspace(-ccol,ccol-1,cols),np.linspace(-crow,crow-1,rows))
        gaussian_filter=np.exp(-(xv**2+yv**2)/(2*sigma**2))
        mask=1-gaussian_filter
        f_shift=f_shift*mask
        f_ishift=np.fft.ifftshift(f_shift)
        img_back=np.fft.ifft2(f_ishift)
        img_back=np.abs(img_back)
        cleaned=cv2.normalize(img_back,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        result=image.copy()
        result[y:y+h,x:x+w]=cleaned
        return result

    def fix_blur(self, image, barcode_region):
        # From codebase 7: single mild unsharp mask
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        gaussian=cv2.GaussianBlur(roi,(0,0),3)
        unsharp=cv2.addWeighted(roi,1.3,gaussian,-0.3,0)
        result=image.copy()
        result[y:y+h,x:x+w]=unsharp
        return result

    def remove_salt_and_pepper(self, image, barcode_region):
        # From codebase 2: iterative median filtering
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        filtered=cv2.medianBlur(roi,3)
        filtered=cv2.medianBlur(filtered,5)
        noise_level=np.std(filtered-roi)
        if noise_level>30:
            filtered=cv2.medianBlur(filtered,3)
        result=image.copy()
        result[y:y+h,x:x+w]=filtered
        return result

    def fix_contrast(self, image, barcode_region):
        # If very low std <5 => aggressive stretch
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        std_val=np.std(roi)
        if std_val<5:
            roi=self.aggressive_stretch(roi)
        else:
            clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            roi=clahe.apply(roi)
            mean_intensity=np.mean(roi)
            # gamma very close to 1
            if mean_intensity<128:
                gamma=1.05
            else:
                gamma=0.95
            invGamma=1.0/gamma
            table=(np.array([(i/255.0)**invGamma*255 for i in range(256)])).astype(np.uint8)
            roi=cv2.LUT(roi,table)
        result=image.copy()
        result[y:y+h,x:x+w]=roi
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

    def remove_foreign_object_after_binarization(self, binary, min_bar_width):
        # Height of the barcode = binary.shape[0]
        # Width = min_bar_width
        h, w = binary.shape
        # We'll scan each column: if too anomalous (e.g., too many white where black expected),
        # replace column by averaging neighboring columns.
        # Criterion: if column deviates from median column intensity?
        col_intensity = np.mean(binary==0, axis=0) # fraction of black pixels
        median_intensity = np.median(col_intensity)
        threshold = median_intensity - 0.2 # If column has 20% less black pixels than median, considered anomalous
        cleaned = binary.copy()
        for i in range(w):
            if col_intensity[i]<threshold:
                # Find neighbors within min_bar_width that are good
                good_cols=[]
                for offset in range(-min_bar_width,min_bar_width+1):
                    ci=i+offset
                    if 0<=ci<w and col_intensity[ci]>=threshold:
                        good_cols.append(cleaned[:,ci])
                if good_cols:
                    avg_col=np.mean(good_cols, axis=0)
                    cleaned[:,i]=(avg_col<127).astype(np.uint8)*255
        return cleaned

    def crop_final(self, binary):
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
        self.add_step("Original Image", image)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image.copy()
        problems, barcode_region=self.detect_image_problems(image)
        self.add_step("Detected Problems: "+", ".join(problems), gray)

        if barcode_region is None:
            # fallback: just binarize entire image
            binary=self.binarize(gray)
            self.add_step("Binarized",binary)
            final=self.crop_final(binary)
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

        # We skip compression detection or rely on a higher threshold
        # If you want, implement compression fix here if confident:
        # no explicit compression problem to reduce false positives

        if 'salt_pepper' in problems:
            processed=self.remove_salt_and_pepper(processed,(x,y,w,h))
            self.add_step("Removed Salt & Pepper",processed)

        if 'blur' in problems:
            processed=self.fix_blur(processed,(x,y,w,h))
            self.add_step("Fixed Blur",processed)

        if 'low_contrast' in problems:
            processed=self.fix_contrast(processed,(x,y,w,h))
            self.add_step("Enhanced Contrast",processed)

        # Binarize ROI
        roi=processed[y:y+h,x:x+w]
        binary=self.binarize(roi)
        self.add_step("Binarized ROI",binary)

        # Foreign object removal after binarization
        # If foreign_object was detected, try final pass
        if 'foreign_object' in problems:
            min_bar_width, _=self.estimate_bar_thickness(roi)
            binary=self.remove_foreign_object_after_binarization(binary, min_bar_width)
            self.add_step("Removed Foreign Object (Post-Binarization)",binary)

        # Final crop
        final=self.crop_final(binary)
        self.add_step("Final Crop",final)
        return final

    def visualize_steps(self):
        n_steps=len(self.steps)
        cols=min(4,n_steps)
        rows=(n_steps+cols-1)//cols
        fig, axes=plt.subplots(rows, cols, figsize=(4*cols,4*rows))
        if rows*cols==1:
            axes=np.array([axes])
        if rows==1:
            axes=axes.reshape(1,-1)
        elif cols==1:
            axes=axes.reshape(-1,1)
        for i,(description,img) in enumerate(self.steps):
            r=i//cols
            c=i%cols
            ax=axes[r,c]
            if len(img.shape)==2:
                ax.imshow(img,cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            ax.set_title(description,fontsize=8)
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
