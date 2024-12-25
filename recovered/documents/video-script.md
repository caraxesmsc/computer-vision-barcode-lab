# Barcode Processing Video Script

## Person 1: Preprocessing (2-3 minutes)
Hello everyone! I'll be explaining the preprocessing steps in our barcode processing system. This is where we prepare the image for barcode detection.

Let's walk through each preprocessing step:

1. First, we convert the image to grayscale:
```python
def turnGrayscale(image: np.ndarray) -> np.ndarray:
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return img_gray
```

2. Then we check for and handle sine wave noise:
```python
def detectSineWaveNoise(image: np.ndarray) -> bool:
    # Using Fourier transform to detect periodic noise
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
```
If sine wave noise is detected, we apply adaptive thresholding to minimize its impact.

3. Next, we enhance image quality through:
```python
def medianFilter(image: np.ndarray) -> np.ndarray:
    img_blur = cv.blur(image, (1, 17))
    img_filtered = cv.medianBlur(img_blur, 5)
```

4. Finally, we improve contrast using histogram stretching:
```python
def histogramStretching(image: np.ndarray) -> np.ndarray:
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = np.uint8((image - min_val) / (max_val - min_val) * 255)
```

These preprocessing steps ensure our image is in the best possible condition for barcode detection. Now, [Person 2] will explain how we handle various edge cases.

## Person 2: Edge Cases (2-3 minutes)
Thanks [Person 1]. I'll be covering how our system handles various challenging situations and edge cases.

1. Foreign Object Detection and Removal:
```python
def detect_foreign_object(image: np.ndarray) -> bool:
    vertical_profile = np.mean(image, axis=0)
    v_std = np.std(vertical_profile)
    v_mean = np.mean(vertical_profile)
    return np.any(np.abs(vertical_profile - v_mean) > 2.5 * v_std)
```

2. When foreign objects are detected, we remove them:
```python
def remove_foreign_object(image: np.ndarray) -> np.ndarray:
    # Remove horizontal objects
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
    # Remove vertical objects
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 25))
```

3. Handling rotation issues:
```python
def rotateBarcode(image: np.ndarray) -> np.ndarray:
    # Find largest contour and its orientation
    rect = cv.minAreaRect(largest_contour)
    angle = rect[-1]
    if angle < -45:
        angle += 90
```

4. Retry mechanism for failed attempts:
```python
def extract_barcode_with_retry(image: np.ndarray, file_name: str, expected_barcode: List[str]):
    # Multiple approaches if standard processing fails
    if detect_foreign_object(preprocessed):
        removed_fo = remove_foreign_object(preprocessed)
    # Try aggressive cropping
    cropped = preprocess_image(image, aggressive_crop=True)
```

Now, [Person 3] will explain the barcode detection process.

## Person 3: Barcode Detection (2-3 minutes)
Thank you [Person 2]. I'll explain how we detect and extract the barcode region from the preprocessed image.

1. Initial barcode extraction:
```python
def extractBarcode(image: np.ndarray, aggressive_crop: bool = False) -> np.ndarray:
    img_thresh_inv = cv.bitwise_not(image)
    x, y, w, h = cv.boundingRect(img_thresh_inv)
```

2. We use two different cropping approaches:
```python
def cropping_pipeline_bounding_rect(binary: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    # Find all external contours
    contours, _ = cv.findContours(inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
```

```python
def cropping_pipeline_projection(binary: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    # Use projections for precise cropping
    vertical_projection = np.sum(binary == 0, axis=0)
    horizontal_projection = np.sum(binary == 0, axis=1)
```

3. Morphological operations to clean up the barcode:
```python
def closeOpen(image: np.ndarray) -> np.ndarray:
    kernel_height = image.shape[0] // 5
    kernel_width = 3
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
```

Over to [Person 4] for the barcode decoding process.

## Person 4: Barcode Decoding (2-3 minutes)
Thanks [Person 3]. I'll explain how we decode the detected barcode into actual numbers.

1. The decoding process starts with our Code 11 pattern dictionary:
```python
def decode_code11_barcode(image: np.ndarray, tolerance: float) -> list:
    code11_widths = {
        "00110": "Stop/Start",
        "10001": "1",
        "01001": "2",
        # ... other patterns
    }
```

2. We analyze the barcode pattern:
```python
    mean_col = image.mean(axis=0)
    mean_col[mean_col <= 127] = 1
    mean_col[mean_col > 128] = 0
```

3. Determine bar sizes and tolerance:
```python
    narrow_bar_size = 0
    for px in pixels:
        if px == "1":
            narrow_bar_size += 1
        else:
            break

    wide_bar_size = narrow_bar_size * 2
    min_narrow = narrow_bar_size * (1 - tolerance)
    max_narrow = narrow_bar_size * (1 + tolerance)
```

4. Finally, we match patterns to decode the numbers:
```python
    while pixel_index < len(pixels):
        if current_digit_widths in code11_widths:
            digits.append(code11_widths[current_digit_widths])
```

The decoded result is then validated against expected patterns and logged:
```python
logger.info(f"Decoded barcode for image {i + 1}: {decoded}")
logger.info(f"Test {i + 1} ({file_name}) => {test_outcome}")
```

And that concludes our explanation of the barcode processing system! Thank you for watching.
