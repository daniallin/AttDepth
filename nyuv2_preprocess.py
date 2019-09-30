from zipfile import ZipFile
import re
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import json


class RawDatasetArchive():
    """Loads a zip file containing (a part of) the raw dataset and
    provides member functions for further data processing.
    """

    def __init__(self, zip_path, extract_path, json_path):
        self.extract_path = extract_path
        self.json_path = json_path
        self.zip_path = zip_path
        self.zip = ZipFile(zip_path)
        self.frames = synchronise_frames(self.zip.namelist())

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def extract_frame(self, frame):
        """Extracts a synchronised frame of depth and color images.

        The frame parameter must be a pair of depth and color maps from
        the archive. Optionally the path of an extraction directory can be given.
        """
        if not os.path.exists(self.extract_path):
            os.mkdir(self.extract_path)

        for f in frame:
            self.zip.extract(f, self.extract_path)

    def get_path(self, invalid_list):
        img_path = list()
        for frame in self.frames:
            rgb_depth = dict()
            depth_path = Path(os.path.join(self.extract_path, frame[0])).as_posix()
            color_path = Path(os.path.join(self.extract_path), frame[1]).as_posix()
            if depth_path in invalid_list or color_path in invalid_list:
                continue
            if not(os.path.exists(depth_path) and os.path.exists(color_path)):
                self.extract_frame(frame)
            rgb_depth['rgb_path'] = color_path
            rgb_depth['depth_path'] = depth_path
            img_path.append(rgb_depth)

        return img_path


def synchronise_frames(frame_names):
    """Constructs a list of synchronised depth and RGB frames.

    Returns a list of pairs, where the first is the path of a depth image,
    and the second is the path of a color image.
    """

    # Regular expressions for matching depth and color images
    depth_img_prog = re.compile(r'.+/d-.+\.pgm')
    color_img_prog = re.compile(r'.+/r-.+\.ppm')

    # Applies a regex program to the list of names
    def match_names(prog):
        return map(prog.match, frame_names)

    # Filters out Nones from an iterator
    def filter_none(iter):
        return filter(None.__ne__, iter)

    # Converts regex matches to strings
    def match_to_str(matches):
        return map(lambda match: match.group(0), matches)

    # Retrieves the list of image names matching a certain regex program
    def image_names(prog):
        return list(match_to_str(filter_none(match_names(prog))))

    depth_img_names = image_names(depth_img_prog)
    color_img_names = image_names(color_img_prog)

    # By sorting the image names we ensure images come in chronological order
    depth_img_names.sort()
    color_img_names.sort()

    def name_to_timestamp(name):
        """Extracts the timestamp of a RGB / depth image from its name."""
        _, time, _ = name.split('-')
        return float(time)

    frames = []
    color_count = len(color_img_names)
    color_idx = 0

    for depth_img_name in depth_img_names:
        depth_time = name_to_timestamp(depth_img_name)
        color_time = name_to_timestamp(color_img_names[color_idx])

        diff = abs(depth_time - color_time)

        # Keep going through the color images until we find
        # the one with the closest timestamp
        while color_idx < color_count-1:
            if color_img_names[color_idx] == "basement_0001c/r-1316653687.842499-3408451313.ppm":
                color_idx = color_idx + 1
                continue
            color_time = name_to_timestamp(color_img_names[color_idx + 1])

            new_diff = abs(depth_time - color_time)

            # Moving forward would only result in worse timestamps
            if new_diff > diff:
                break
            diff = new_diff

            color_idx = color_idx + 1

        frames.append((depth_img_name, color_img_names[color_idx]))

    return frames


def load_depth_image(path):
    """Loads an unprocessed depth map extracted from the raw dataset."""
    with open(path, 'rb') as f:
        return Image.fromarray(read_pgm(f), mode='I')

def load_color_image(path):
    """Loads an unprocessed color image extracted from the raw dataset."""
    with open(path, 'rb') as f:
        return Image.fromarray(read_ppm(f), mode='RGB')

def read_pgm(pgm_file):
    """Reads a PGM file from a buffer.

    Returns a numpy array of the appropiate size and dtype.
    """

    # First line contains some image meta-info
    p5, width, height, depth = pgm_file.readline().split()

    # Ensure we're actually reading a P5 file
    assert p5 == b'P5'
    assert depth == b'65535', "Only 16-bit PGM files are supported"

    width, height = int(width), int(height)

    data = np.fromfile(pgm_file, dtype='<u2', count=width*height)

    return data.reshape(height, width).astype(np.uint32)

def read_ppm(ppm_file):
    """Reads a PPM file from a buffer.

    Returns a numpy array of the appropiate size and dtype.
    """

    p6, width, height, depth = ppm_file.readline().split()

    assert p6 == b'P6'
    assert depth == b'255', "Only 8-bit PPM files are supported"

    width, height = int(width), int(height)

    data = np.fromfile(ppm_file, dtype=np.uint8, count=width*height*3)

    return data.reshape(height, width, 3)


MAX_DEPTH = 10.0

def depth_rel_to_depth_abs(depth_rel):
    """Projects a depth image from internal Kinect coordinates to world coordinates.

    The absolute 3D space is defined by a horizontal plane made from the X and Z axes,
    with the Y axis pointing up.

    The final result is in meters."""

    DEPTH_PARAM_1 = 351.3
    DEPTH_PARAM_2 = 1092.5

    depth_abs = DEPTH_PARAM_1 / (DEPTH_PARAM_2 - depth_rel)

    return np.clip(depth_abs, 0, MAX_DEPTH)


def color_depth_overlay(color, depth_abs, relative=False):
    """Overlay the depth of a scene over its RGB image to help visualize
    the alignment.

    Requires the color image and the corresponding depth map. Set the relative
    argument to true if the depth map is not already in absolute depth units
    (in meters).

    Returns a new overlay of depth and color.
    """

    assert color.size == depth_abs.size, "Color / depth map size mismatch"

    depth_arr = np.array(depth_abs).astype(np.float32)

    if relative == True:
        depth_arr = depth_rel_to_depth_abs(depth_arr)

    depth_ch = (depth_arr - np.min(depth_arr)) / np.max(depth_arr)
    depth_ch = (depth_ch * 255).astype(np.uint8)
    depth_ch = Image.fromarray(depth_ch)

    r, g, _ = color.split()

    return Image.merge("RGB", (r, depth_ch, g)), color, depth_ch

def plot_color(ax, color, title="Color"):
    """Displays a color image from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)


if __name__ == '__main__':
    img_path = list()
    zip_root = '/home/root123/Datasets/NYUv2'

    with open('dataset/invalid.txt', 'r') as f:
        lines = f.readlines()
        new_lines = list(set(lines[:-1]))
        for i, l in enumerate(new_lines):
            new_lines[i] = new_lines[i].split('\n')[0]

    for z in os.listdir(zip_root):
        if '.zip' not in z:
            continue
        zip_path = os.path.join(zip_root, z)
        raw_archive = RawDatasetArchive(zip_path,
                                        extract_path='/home/root123/workspace/nyuv2_extract',
                                        json_path='dataset/nyuv2')
        img_path_child = raw_archive.get_path(new_lines)
        img_path.extend(img_path_child)

    random.shuffle(img_path)
    img_num = len(img_path)
    with open("dataset/nyuv2/train_raw_annotations.json", "w") as f:
        json.dump(img_path[:int(img_num * 0.9)], f)
    with open("dataset/nyuv2/val_raw_annotations.json", "w") as f:
        json.dump(img_path[int(img_num * 0.9):], f)

    # frame = raw_archive[5]
    # depth_path, color_path = Path('../dataset/nyuv2_extract') / frame[0], Path('../dataset/nyuv2_extract') / frame[1]
    #
    # if not (depth_path.exists() and color_path.exists()):
    #     raw_archive.extract_frame(frame)
    #
    # # color = load_color_image(os.path.join('E:/DepthEstimation/Champion/dataset/nyu_extract', color_path))
    # # depth = load_depth_image(os.path.join('E:/DepthEstimation/Champion/dataset/nyu_extract', depth_path))
    # color = load_color_image(color_path)
    # depth = load_depth_image(depth_path)
    #
    # fig = plt.figure("Raw Dataset Sample", figsize=(12, 5))
    #
    # before_proj_overlay = color_depth_overlay(color, depth, relative=True)
    #
    # ax = fig.add_subplot(2, 2, 1)
    # plot_color(ax, before_proj_overlay[0], "Before Projection")
    #
    # ax = fig.add_subplot(2, 2, 3)
    # plot_color(ax, before_proj_overlay[1], "Before Projection")
    # ax = fig.add_subplot(2, 2, 4)
    # plot_color(ax, before_proj_overlay[2], "Before Projection")
    #
    # plt.show()


