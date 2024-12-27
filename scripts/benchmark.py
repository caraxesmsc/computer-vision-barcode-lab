"""Run the recovered version89 variant against its original labelled images."""
import argparse
import ast
import importlib.util
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'src' / 'barcode_pipeline.py'


def expected_values():
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'expected_barcodes'
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError('Expected values missing from recovered source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT / 'data' / 'test-cases')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('version89_variant', SOURCE)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected = expected_values()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    by_number = {}
    for path in paths:
        prefix = path.name.split(' ', 1)[0]
        if prefix.isdigit() and 1 <= int(prefix) <= len(expected):
            index = int(prefix) - 1
            if index in by_number:
                parser.error(f'duplicate case number {prefix}')
            by_number[index] = path
    if len(by_number) != len(expected):
        parser.error(f'expected numbered cases 01–{len(expected):02}; found {len(by_number)}')
    passed = 0
    for index, path in sorted(by_number.items()):
        image = cv2.imread(str(path))
        try:
            if image is None:
                raise ValueError('could not read image')
            _, decoded = module.extract_barcode_with_retry(image, path.name, expected[index])
            actual = ''.join(d for d in decoded if d != 'Stop/Start')
            wanted = ''.join(d for d in expected[index] if d != 'Stop/Start')
            ok = actual == wanted
            passed += ok
            print(f'{path.name}: {"PASS" if ok else "FAIL"} (decoded={actual!r}, expected={wanted!r})', flush=True)
        except Exception as exc:
            print(f'{path.name}: ERROR ({type(exc).__name__}: {exc})', flush=True)
    print(f'Result: {passed}/{len(expected)} matched (known-answer retry benchmark)')
    return 0 if passed == len(expected) else 1


if __name__ == '__main__':
    raise SystemExit(main())
