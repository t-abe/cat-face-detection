import sys
import numpy as np
from skimage import io, transform
from glob import glob
import random

def main():
    if len(sys.argv) < 4:
        print "./crop.faces.py INPUT_DIR OUTPUT_DIR N"
        return
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    n_negatives = int(sys.argv[3])

    image_list = glob('%s/*.jpg' % input_dir) + glob('%s/*.png' % input_dir)
    count = 0
    for i, image_path in enumerate(image_list):
        image = io.imread(image_path)
        assert image.shape[0] >= 64 or image.shape[1] >= 64
        while n_negatives * (i + 1) / len(image_list) > count:
            cropped = crop_randomly(image)
            io.imsave('%s/%d.png' % (output_dir, count), cropped)
            count += 1

def crop_randomly(image):
    h, w, _ = image.shape
    x = random.randint(0, w - 64)
    y = random.randint(0, h - 64)
    cropped = image[y:y + 64, x:x + 64]
    return cropped

if __name__ == "__main__":
    main()
