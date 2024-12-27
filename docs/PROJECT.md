# Project documentation

## Problem and context

This Fall 2024 coursework investigated preprocessing and decoding Code 11 barcodes in a small supplied set of 11 images. Images range from clean examples to blur, low or uneven contrast, salt-and-pepper noise, compression, rotation, an obstruction, and a combination of frequency interference and rotation. The difficulty is preserving the relative widths of bars and spaces while correcting the image: filtering or cropping can also destroy the features needed for decoding.

## Recovered pipeline

The local benchmark executes `src/barcode_pipeline.py`. This is one of two source snapshots recovered from the team's materials (the other is in `archive/recovered/`); the available evidence does not establish which one was the final version. The snapshots are preserved as supplied and should be read as historical experiments.

1. **Grayscale and noise heuristic.** `turnGrayscale` converts a color image into one channel. `detectSineWaveNoise` analyzes FFT magnitude peaks; if its heuristic reports periodic noise, the code applies adaptive thresholding. This heuristic is not a general guarantee of noise detection or removal.
2. **Orientation.** `rotateBarcode` finds edges and contours, estimates orientation from the largest contour's minimum-area rectangle, and rotates the image. The estimate can fail when the largest contour is not the barcode.
3. **Filtering and region extraction.** `medianFilter` applies blur, a median filter, and Otsu thresholding. `histogramStretching` stretches intensities when the range is nonzero. `extractBarcode` selects a bounding region; `closeOpen` applies morphology.
4. **Decode.** `decode_code11_barcode` averages pixels by column, thresholds this profile, estimates narrow and wide runs, and matches sequences to a Code 11 lookup table. The implementation makes assumptions about run widths and does not implement checksum validation.
5. **Retry.** `extract_barcode_with_retry` first processes and decodes the image, then may try foreign-object removal, aggressive cropping, and their combination. The retry logic compares each result with the expected payload provided by the caller. Its final result may omit start/stop markers, so the benchmark compares payloads consistently.

The other recovered version implements a different crop/retry strategy. The archived presentation script describes ideas and functions across versions, so it should not be treated as a precise specification of the variant.

## Running the benchmark

Follow the setup in the [README](../README.md#get-started). The runner locates images by their leading case numbers, rather than relying on directory enumeration order. It reads expected sequences from the recovered source, removes `Stop/Start` markers for comparison, prints individual outcomes, and returns exit code 1 if any case fails. Missing or duplicate case numbers are errors. It catches processing exceptions per image so that the remaining cases can still be examined.

## Results and limitations

A local run using Python 3.14 on the supplied test images produced matching payloads for cases 01, 02, and 04–11; case 03 returned an empty payload. Dependencies are not pinned, so results may vary across environments. The benchmark uses the known expected answer to select retries, making it unsuitable for estimating accuracy on unseen inputs. It does not verify Code 11 checksums, and cannot demonstrate that the recovered image retains all original information. No live or production readiness is claimed.

## Provenance and team

The original repository preserved a series of earlier experiments. Two later `version89` files, the image archive, and the presentation script were recovered from materials supplied by the team. The variant used by the benchmark is in `src/`; the other snapshot and presentation script are in `archive/recovered/`. The 11 images have had metadata removed; their pixel content is retained. Reports containing student IDs were not added. The earlier experiments and output images have been organized into `archive/`, and paths assumed by old scripts may need adjustment.

The team members are Mohamed Hossam, Youssef Kassab, Monica Hany, and Patrick Ramez. The recovered project commit carries their co-author credits. The project materials identify Kassab with foreign-object and cropping snippets, but do not establish function-level authorship across the entire project.

## Ideas for further work

Separate inference from benchmark-only expected answers; define explicit decode validity and checksum handling; evaluate on independent images with known ground truth; characterize failure cases, including obstruction; and add reproducible dependency versions and targeted regression tests. These are proposed improvements, not features currently implemented.
