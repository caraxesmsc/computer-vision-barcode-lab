import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

class BarcodePreprocessor:
    def __init__(self):
        self.steps = []
        self.problems = []

    def add_step(self, description, image):
        self.steps.append((description, image.copy()))

    def aggressive_stretch(self, img):
        min_val, max_val = np.min(img), np.max(img)
        if max_val == min_val:
            return img
        return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    def estimate_bar_thickness(self, image):
        # Binarize for bar estimation
        temp_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 61, 10)
        heights = []
        for col in range(temp_bin.shape[1]):
            column = temp_bin[:, col]
            run_length = 0
            black_runs = []
            for val in column:
                if val == 0:
                    run_length += 1
                else:
                    if run_length > 0:
                        black_runs.append(run_length)
                    run_length = 0
            if run_length>0:
                black_runs.append(run_length)
            if black_runs:
                heights.extend(black_runs)
        if not heights:
            return 5
        return int(np.median(heights))

    def detect_barcode_region(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(sobelx)
        blur = cv2.GaussianBlur(gradx, (21,21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            aspect = w/h
            roi = gradx[y:y+h, x:x+w]
            edge_density = np.sum(roi>50)/(w*h)
            if 1.5 < aspect < 5 and edge_density>0.1:
                valid_regions.append((x,y,w,h))
        if not valid_regions:
            return None
        return max(valid_regions, key=lambda r: np.sum(gradx[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))

    def detect_compression_artifacts(self, gray):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0,ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        block_size = 8
        blurred = cv2.boxFilter(magnitude.astype(np.float32), -1, (block_size,block_size))
        block_variation = np.std(blurred)
        return block_variation > 12  # slightly stricter threshold

    def detect_image_problems(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)>2 else image
        barcode_region = self.detect_barcode_region(gray)
        if barcode_region is None:
            return [], (0,0,gray.shape[1],gray.shape[0])

        x,y,w,h = barcode_region
        roi = gray[y:y+h, x:x+w]

        p5, p95 = np.percentile(roi, [5,95])
        low_contrast = (p95-p5) < 50

        f_transform = np.fft.fft2(roi)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20*np.log(np.abs(f_shift)+1)
        freq_problem = np.std(magnitude_spectrum) > 50

        # Bar thickness for foreign object detection (no freq or contrast fix yet)
        # To keep it simple, just estimate directly
        bar_thickness = self.estimate_bar_thickness(roi)

        # Foreign object detection
        vertical_profile = np.mean(roi, axis=0)
        horizontal_profile = np.mean(roi, axis=1)
        v_std = np.std(vertical_profile)
        h_std = np.std(horizontal_profile)
        v_anomalies = np.where(np.abs(vertical_profile - np.mean(vertical_profile))>3.0*v_std)[0]
        h_anomalies = np.where(np.abs(horizontal_profile - np.mean(horizontal_profile))>3.0*h_std)[0]

        problems = []
        if len(v_anomalies)>bar_thickness or len(h_anomalies)>bar_thickness:
            problems.append('foreign_object')

        # Blur check: Laplacian var + edge density
        lap_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        edges = cv2.Canny(roi, 50,150)
        edge_density = np.mean(edges>0)
        if lap_var<90 and edge_density<0.15:
            problems.append('blur')

        # Low contrast
        if low_contrast:
            problems.append('low_contrast')

        # Salt & Pepper
        median = cv2.medianBlur(roi,3)
        noise_diff = np.abs(roi-median)
        # Stricter threshold
        if np.percentile(noise_diff,95)>40:
            problems.append('salt_pepper')

        # Compression artifacts
        if self.detect_compression_artifacts(roi):
            problems.append('compression_artifacts')

        # Frequency domain
        if freq_problem:
            problems.append('frequency_domain')

        # Rotation
        lines = cv2.HoughLines(edges,1,np.pi/180,threshold=min(w,h)//3)
        if lines is not None:
            angles=[]
            for rho,theta in lines[:,0]:
                angle = np.degrees(theta)
                if angle<45:
                    angles.append(angle)
                elif angle>135:
                    angles.append(angle-180)
            if angles and abs(np.median(angles))>5:
                problems.append('rotation')

        return problems, barcode_region

    def get_anomalous_regions(self, anomalies, bar_thickness):
        regions=[]
        if len(anomalies)==0:
            return regions
        current=[anomalies[0]]
        for i in range(1,len(anomalies)):
            if anomalies[i]==anomalies[i-1]+1 and (anomalies[i]-current[0])<bar_thickness*2:
                current.append(anomalies[i])
            else:
                regions.append(current)
                current=[anomalies[i]]
        regions.append(current)
        return regions

    def fix_frequency_domain_simple(self, image):
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        sigma=30
        x=np.linspace(-ccol, ccol-1, cols)
        y=np.linspace(-crow, crow-1, rows)
        xv,yv=np.meshgrid(x,y)
        gaussian_filter=np.exp(-(xv**2+yv**2)/(2*sigma**2))
        mask=1-gaussian_filter
        f_shift=f_shift*mask
        f_ishift=np.fft.ifftshift(f_shift)
        img_back=np.fft.ifft2(f_ishift)
        img_back=np.abs(img_back)
        return cv2.normalize(img_back,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    def fix_compression_artifacts(self, image):
        return cv2.fastNlMeansDenoising(image,None,5,7,21)

    def remove_salt_pepper(self, image, barcode_region):
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

    def fix_blur(self, image, barcode_region):
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        gaussian=cv2.GaussianBlur(roi,(0,0),2)
        unsharp=cv2.addWeighted(roi,1.5,gaussian,-0.5,0)
        lap_var=cv2.Laplacian(unsharp,cv2.CV_64F).var()
        if lap_var<150:
            kernel=np.array([[-1,-1,-1],
                              [-1,9,-1],
                              [-1,-1,-1]])
            unsharp=cv2.filter2D(unsharp,-1,kernel)
        result=image.copy()
        result[y:y+h,x:x+w]=unsharp
        return result

    def fix_contrast(self, image, barcode_region):
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        enhanced=clahe.apply(roi)
        mean_intensity=np.mean(enhanced)
        if mean_intensity<128:
            gamma=1.5
        else:
            gamma=0.7
        invGamma=1.0/gamma
        table=(np.array([((i/255.0)**invGamma)*255 for i in range(256)])).astype("uint8")
        enhanced=cv2.LUT(enhanced, table)
        result=image.copy()
        result[y:y+h,x:x+w]=enhanced
        return result

    def remove_foreign_object(self, image, barcode_region):
        x,y,w,h=barcode_region
        roi=image[y:y+h,x:x+w].copy()
        bar_thickness=self.estimate_bar_thickness(roi)

        vertical_profile=np.mean(roi,axis=0)
        horizontal_profile=np.mean(roi,axis=1)
        v_std=np.std(vertical_profile)
        h_std=np.std(horizontal_profile)
        v_anomalies=np.where(np.abs(vertical_profile-np.mean(vertical_profile))>3*v_std)[0]
        h_anomalies=np.where(np.abs(horizontal_profile-np.mean(horizontal_profile))>3*h_std)[0]
        v_regions=self.get_anomalous_regions(v_anomalies, bar_thickness)
        h_regions=self.get_anomalous_regions(h_anomalies, bar_thickness)

        cleaned=roi.copy()

        # vertical anomalies correction
        for region in v_regions:
            for i in region:
                left_vals=[]
                right_vals=[]
                for j in range(max(0,i-bar_thickness),i):
                    if j>=0 and j<w and not (j in v_anomalies):
                        left_vals.append(roi[:,j])
                for j in range(i+1,min(w,i+bar_thickness+1)):
                    if j<w and not (j in v_anomalies):
                        right_vals.append(roi[:,j])
                if left_vals and right_vals:
                    cleaned[:,i]=np.mean([left_vals[-1],right_vals[0]],axis=0)
                elif left_vals:
                    cleaned[:,i]=left_vals[-1]
                elif right_vals:
                    cleaned[:,i]=right_vals[0]

        # horizontal anomalies correction
        for region in h_regions:
            for j in region:
                top_vals=[]
                bottom_vals=[]
                for i_ in range(max(0,j-bar_thickness),j):
                    if i_>=0 and i_<h and not (i_ in h_anomalies):
                        top_vals.append(cleaned[i_,:])
                for i_ in range(j+1,min(h,j+bar_thickness+1)):
                    if i_<h and not (i_ in h_anomalies):
                        bottom_vals.append(cleaned[i_,:])
                if top_vals and bottom_vals:
                    cleaned[j,:]=np.mean([top_vals[-1],bottom_vals[0]],axis=0)
                elif top_vals:
                    cleaned[j,:]=top_vals[-1]
                elif bottom_vals:
                    cleaned[j,:]=bottom_vals[0]

        result=image.copy()
        result[y:y+h,x:x+w]=cleaned
        return result

    def fix_rotation(self, image):
        # rotation first with padding
        pad=int(max(image.shape)*0.2)
        padded=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=255)
        edges=cv2.Canny(padded,50,150)
        lines=cv2.HoughLines(edges,1,np.pi/180,100)
        if lines is not None:
            angles=[]
            for rho,theta in lines[:,0]:
                angle=np.degrees(theta)
                if angle<45:
                    angles.append(angle)
                elif angle>135:
                    angles.append(angle-180)
            if angles:
                median_angle=np.median(angles)
                rows,cols=padded.shape
                M=cv2.getRotationMatrix2D((cols/2,rows/2),median_angle,1)
                rotated=cv2.warpAffine(padded,M,(cols,rows),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,value=255)
                coords=np.column_stack(np.where(rotated<250))
                if len(coords)>0:
                    x,y,w,h=cv2.boundingRect(coords)
                    rotated=rotated[y:y+h,x:x+w]
                return rotated
        return image

    def binarize(self, image):
        p2,p98=np.percentile(image,(2,98))
        if p2==p98:
            normalized=image
        else:
            normalized=np.interp(image,(p2,p98),(0,255)).astype(np.uint8)
        binary=cv2.adaptiveThreshold(
            normalized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5
        )
        kernel=np.ones((2,2),np.uint8)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        return binary

    def final_crop(self, binary):
        # After binarization and rotation, final tight crop including numbers beneath
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary
        largest=max(contours,key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(largest)
        # Add padding to ensure numbers beneath included
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
        problems,barcode_region=self.detect_image_problems(image)
        self.add_step("Detected Problems: "+", ".join(problems), gray)

        processed=gray.copy()

        # Rotation first if needed
        if 'rotation' in problems:
            rotated=self.fix_rotation(processed)
            self.add_step("Fixed Rotation",rotated)
            processed=rotated
        else:
            self.add_step("No Rotation Needed",processed)

        # Frequency domain after rotation if needed
        if 'frequency_domain' in problems:
            processed=self.fix_frequency_domain_simple(processed)
            self.add_step("Fixed Frequency Domain",processed)

        # Foreign object now
        _, updated_region=self.detect_image_problems(cv2.cvtColor(processed,cv2.COLOR_GRAY2BGR))
        if not updated_region:
            updated_region=barcode_region
        if 'foreign_object' in problems:
            processed=self.remove_foreign_object(processed, updated_region)
            self.add_step("Removed Foreign Object",processed)

        # Compression artifacts
        if 'compression_artifacts' in problems:
            processed=self.fix_compression_artifacts(processed)
            self.add_step("Fixed Compression Artifacts",processed)

        # Salt & Pepper
        if 'salt_pepper' in problems:
            _, updated_region=self.detect_image_problems(cv2.cvtColor(processed,cv2.COLOR_GRAY2BGR))
            if not updated_region:
                updated_region=barcode_region
            processed=self.remove_salt_pepper(processed,updated_region)
            self.add_step("Removed Salt & Pepper",processed)

        # Blur
        if 'blur' in problems:
            _, updated_region=self.detect_image_problems(cv2.cvtColor(processed,cv2.COLOR_GRAY2BGR))
            if not updated_region:
                updated_region=barcode_region
            processed=self.fix_blur(processed,updated_region)
            self.add_step("Fixed Blur",processed)

        # Low contrast
        if 'low_contrast' in problems:
            _, updated_region=self.detect_image_problems(cv2.cvtColor(processed,cv2.COLOR_GRAY2BGR))
            if not updated_region:
                updated_region=barcode_region
            processed=self.fix_contrast(processed,updated_region)
            self.add_step("Enhanced Contrast",processed)

        # Binarize
        binary=self.binarize(processed)
        self.add_step("Binarized", binary)

        # Final Crop including numbers beneath
        cropped=self.final_crop(binary)
        self.add_step("Cropped Barcode",cropped)

        return cropped

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

if __name__ == "__main__":
    preprocessor=BarcodePreprocessor()
    image_path=filedialog.askopenfilename(title="Select an Image",filetypes=[("Image Files","*.png;*.jpg;*.jpeg")])
    if image_path:
        image=cv2.imread(image_path)
        if image is not None:
            processed_image=preprocessor.preprocess_image(image)
            preprocessor.visualize_steps()
