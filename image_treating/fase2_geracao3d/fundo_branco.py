import argparse

import numpy as np
from PIL import Image
from scipy.ndimage import label

WHITE_MIN = 240  # canal > 240 em R, G e B = "branco"


def background_mask(rgb):
    """Pixels brancos conectados a algum canto (branco dentro do dente não é fundo)."""
    regions, _ = label(np.all(rgb > WHITE_MIN, axis=2))
    corners = {regions[0, 0], regions[0, -1], regions[-1, 0], regions[-1, -1]} - {0}
    return np.isin(regions, list(corners))


def remove_bg(input_path, output_path):
    arr = np.array(Image.open(input_path).convert("RGBA"))
    arr[background_mask(arr[:, :, :3]), 3] = 0
    Image.fromarray(arr).save(output_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    remove_bg(args.input, args.output)


if __name__ == "__main__":
    main()
