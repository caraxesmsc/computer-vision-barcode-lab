# CV Playground: Code 11 barcode project

A Fall 2024 computer vision coursework project exploring image preprocessing and Code 11 barcode decoding under difficult image conditions. The original repository contains earlier experiments; `recovered/` contains material found later in the team's project traces.

## Team

- Mohamed Hossam ([caraxesmsc](https://github.com/caraxesmsc))
- Youssef Ayman Mohamed Kassab ([yakassab](https://github.com/yakassab))
- Monica Hany ([Monica-hany](https://github.com/Monica-hany))
- Patrick Ramez ([Patrick-ramez](https://github.com/Patrick-ramez))

Names are supported by the supplied phase reports. The WhatsApp notes mark foreign-object detection, removal and cropping snippets as Kassab's; the other snippets and presentation assignments do not establish function-level authorship. Existing Git history attributes commits to one account, so this roster records the team without inventing commit history.

## Recovered material

- `recovered/source/version89.py` and `version89_variant.py` are verbatim copies of two supplied versions. Their names do not establish which is newer or final; the variant includes a different cropping/retry implementation.
- `recovered/test-cases/` contains the 11 supplied images. The source uses known expected answers for those images.
- `recovered/documents/` contains the supplied presentation script. The phase reports were left out because they contain student IDs. The script describes intended behavior and should not be treated as independent validation of accuracy.
- Earlier experiments and outputs remain in their original locations.

## Run the recovered benchmark

Python, pip, and a Python installation with `tkinter` are required (the recovered script imports it even for the headless benchmark).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r recovered/requirements.txt
python recovered/benchmark.py
```

The runner matches images to expected results by their numbered filenames, reports errors per case, compares the decoded payload without start/stop markers, and exits nonzero if any case fails. `--images PATH` selects a different folder containing the same numbered cases.

This is a **known-answer benchmark**: the historical retry function receives the expected barcode and uses it to select a result. It does not demonstrate general decoding of unknown images or validate Code 11 checksums. The recovered source has not been modernized and may contain bugs. The two variants should be treated as historical snapshots until the team identifies the final version and verifies the original results.

## Contribution credit

Team credit is recorded above. For GitHub's contributor and contribution displays, use real authored commits or `Co-authored-by` trailers on work each person actually contributed to, with an email associated with their GitHub account. The teammates can also be invited as repository collaborators. Their participation and any retrospective attribution should be confirmed with them; the traces do not establish who wrote every function.

On the recovered images, the variant produced 10 of 11 matching payloads in a local run using Python 3.14. Case 03 (foreign object over the barcode) returned an empty payload. This is a local baseline, not a guarantee of performance on other images or dependency versions.
