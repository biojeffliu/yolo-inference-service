# yolo_service/scripts/binary_to_png.py
# Takes JSON masks and converts them to binary PNG files.

# Note: this script does not consider multiple objects in a single mask.

import json
import os
import cv2
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

def _to_numpy_mask(candidate: Any) -> np.ndarray:
    """
    Convert a JSON-loaded object (list of lists / booleans / ints) to a 2D numpy array.
    Raises ValueError if not a proper 2D rectangular list.
    """
    if isinstance(candidate, np.ndarray):
        arr = candidate
    else:
        arr = np.array(candidate, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Mask is not 2D (ndim={arr.ndim}).")

    # Binarize: anything > 0 becomes 255, else 0
    arr = (arr > 0).astype(np.uint8) * 255
    return arr

def _save_mask(mask: np.ndarray, out_path: str) -> None:
    # Ensure uint8 single-channel
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Write PNG
    ok = cv2.imwrite(out_path, mask)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {out_path}")

def _extract_masks(data: Any) -> List[Tuple[str, np.ndarray]]:
    """
    Extract one or more (name, mask) pairs from various JSON shapes.
    Returns a list of (name, np.ndarray) where name is a short identifier per mask.
    """
    masks: List[Tuple[str, np.ndarray]] = []

    # Case A: the whole file is a single 2D mask (list of lists)
    if isinstance(data, list):
        try:
            masks.append(("mask", _to_numpy_mask(data)))
            return masks
        except Exception as e:
            raise ValueError(f"Top-level list didn't parse as a 2D mask: {e}")

    # Case B: dict-based structures
    if isinstance(data, dict):
        # Common wrapper key
        if "masks" in data:
            m = data["masks"]
            if isinstance(m, dict):
                for k, v in m.items():
                    masks.append((str(k), _to_numpy_mask(v)))
                return masks
            elif isinstance(m, list):
                # Either list of 2D masks, or a single 2D mask
                # Name them by index
                for i, v in enumerate(m):
                    masks.append((f"mask{i:03d}", _to_numpy_mask(v)))
                return masks
            else:
                raise ValueError("`masks` key is neither dict nor list.")

        # Otherwise, treat each value in the dict as a mask candidate
        # (e.g., {"1": [[...]], "2": [[...]]})
        added = False
        for k, v in data.items():
            # Skip non-list entries
            if isinstance(v, list):
                try:
                    masks.append((str(k), _to_numpy_mask(v)))
                    added = True
                except Exception:
                    # Not a mask; ignore this key
                    pass
        if added:
            return masks

        # If we reach here and didn't return, try a last-resort:
        # maybe the dict itself is a 2D structure with numeric keys?
        # (Uncommon; skip to avoid false positives)
        raise ValueError("No 2D mask-like lists found in dict values.")

    raise ValueError("Unsupported JSON shape for masks.")

def convert_masks_to_png(input_dir: str, output_dir: str, overwrite: bool = True):
    """Convert binary masks in JSON files to PNG images.

    Args:
        input_dir (str): Directory containing input JSON files with binary masks.
        output_dir (str): Directory to save the converted PNG files.
        overwrite (bool): Whether to overwrite existing PNGs with the same name.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    if not json_files:
        print(f"[WARN] No .json files found in {input_dir}")
        return

    for filename in json_files:
        json_path = os.path.join(input_dir, filename)
        base = os.path.splitext(filename)[0]

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {json_path}: {e}")
            continue

        try:
            named_masks = _extract_masks(data)
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            continue

        if not named_masks:
            print(f"[WARN] {filename}: No masks extracted.")
            continue

        for name, mask in named_masks:
            out_name = f"{base}_{name}.png" if len(named_masks) > 1 else f"{base}.png"
            out_path = os.path.join(output_dir, out_name)

            if (not overwrite) and os.path.exists(out_path):
                print(f"[INFO] Skipping existing {out_path}")
                continue

            try:
                _save_mask(mask, out_path)
                h, w = mask.shape
                print(f"[OK] Wrote {out_path} ({w}x{h})")
            except Exception as e:
                print(f"[ERROR] Failed to write {out_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert binary masks in JSON to PNG files.")
    parser.add_argument("input_dir", type=str, help="Path to input JSON directory with binary masks")
    parser.add_argument("output_dir", type=str, help="Directory to save PNG files")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing PNGs")
    args = parser.parse_args()

    convert_masks_to_png(args.input_dir, args.output_dir, overwrite=not args.no_overwrite)

if __name__ == "__main__":
    main()
