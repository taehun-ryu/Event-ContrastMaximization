import os
import glob
from datetime import datetime
import cv2
import argparse
import re

def find_latest_output_dir(base_dir="../output"):
    subdirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError("No output directory found.")
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir

def numerical_sort_key(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')


def load_and_show_images(output_dir):
    image_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {output_dir}")

    image_paths.sort(key=numerical_sort_key)
    
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load: {path}")
            continue
        cv2.imshow("IWE Viewer", img)
        print(f"Showing: {os.path.basename(path)}")
        key = cv2.waitKey(250)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_path", default=None, help="Output directory path"
    )
    args = parser.parse_args()
    dir_path = args.dir_path or find_latest_output_dir()
    print(f"Loading images from: {dir_path}")
    load_and_show_images(dir_path)
