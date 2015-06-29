import sys
import numpy as np
from skimage import io, transform
from glob import iglob

def main():
    if len(sys.argv) < 3:
        print "./crop.faces.py INPUT_DIR OUTPUT_DIR"
        return
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    for i, image_path in enumerate(iglob('%s/*/*.jpg' % input_dir)):
        annotation_path = '%s.cat' % image_path
        try:
            annotation = parse_annotation(open(annotation_path).read())
        except:
            continue
        face = crop_face(io.imread(image_path), annotation)
        if face != None:
            io.imsave('%s/%d.png' % (output_dir, i), face)

def parse_annotation(line):
    v = map(int, line.split())
    ret = {}
    parts = ["left_eye", "right_eye", "mouth",
             "left_ear1", "left_ear2", "left_ear3",
             "right_ear1", "right_ear2", "right_ear3"]
    for i, part in enumerate(parts):
        if i >= v[0]: break
        ret[part] = np.array([v[1 + 2 * i], v[1 + 2 * i + 1]])
    return ret

def crop_face(image, an):
    diff_eyes = an["left_eye"] - an["right_eye"]
    if diff_eyes[0] == 0 or abs(float(diff_eyes[1]) / diff_eyes[0]) > 0.5:
        return None
    center = (an["left_eye"] + an["right_eye"] + an["mouth"]) / 3
    if center[1] > an["mouth"][1]: return None
    radius = np.linalg.norm(diff_eyes) * 1.1
    xu = center[0] - radius
    xl = center[0] + radius
    yu = center[1] - radius
    yl = center[1] + radius
    if xu < 0 or yu < 0: return None
    if xl > image.shape[1] or yl > image.shape[0]:
        return None
    cropped = image[yu:yl, xu:xl]
    return transform.resize(cropped, (64, 64))

if __name__ == "__main__":
    main()
