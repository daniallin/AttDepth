import json
import os
import numpy as np


class NYUV2Raw():
    def __init__(self, file_path, phase='train'):
        self.phase = phase
        self.dir_anno = os.path.join(file_path, phase + '_raw_annotations.json')
        self.rgb_paths, self.depth_paths = self.get_paths()
        self.uniform_size = (480, 640)

    def get_paths(self):
        with open(self.dir_anno, 'r') as load_f:
            paths = json.load(load_f)
        self.data_size = len(paths)

        rgb_paths = [paths[i]['rgb_path'] for i in range(len(paths))]
        depth_path = [paths[i]['depth_path'] for i in range(len(paths))]

        return rgb_paths, depth_path

    def __len__(self):
        return self.data_size

    def get_data(self):
        invalid_rgb = set()
        invalid_depth = set()
        for index in range(self.data_size):
            rgb_path = self.rgb_paths[index]
            depth_path = self.depth_paths[index]
            # print(rgb_path)
            try:
                with open(rgb_path, 'rb') as r:
                    self.read_ppm(r)  # bgr, H*W*C
            except:
                invalid_rgb.add(rgb_path)

            try:
                with open(depth_path, 'rb') as d:
                    self.read_pgm(d)
            except:
                invalid_depth.add(depth_path)
        return invalid_rgb, invalid_depth

    def read_pgm(self, pgm_file):
        """Reads a PGM file from a buffer.

        Returns a numpy array of the appropiate size and dtype.
        """

        # First line contains some image meta-info
        p5, width, height, depth = pgm_file.readline().split()

        # Ensure we're actually reading a P5 file
        assert p5 == b'P5'
        assert depth == b'65535', "Only 16-bit PGM files are supported"

        width, height = int(width), int(height)

        data = np.fromfile(pgm_file, dtype='<u2', count=width * height)

        data.reshape(height, width).astype(np.uint32)

    def read_ppm(self, ppm_file):
        """Reads a PPM file from a buffer.

        Returns a numpy array of the appropiate size and dtype.
        """

        p6, width, height, depth = ppm_file.readline().split()

        assert p6 == b'P6'
        assert depth == b'255', "Only 8-bit PPM files are supported"

        width, height = int(width), int(height)

        data = np.fromfile(ppm_file, dtype=np.uint8, count=width * height * 3)

        data.reshape(height, width, 3)


if __name__ == '__main__':
    # nyu = NYUV2Raw('/home/root123/workspace/AttDepth/dataset/nyuv2', phase='val')
    # invalid_rgb, invalid_depth = nyu.get_data()

    invalid_rgb, invalid_depth = list(), list()
    with open('invalid.txt', 'r') as f:
        lines = f.readlines()
        new_lines = list(set(lines))
        for l in new_lines:
            if 'd-' in l:
                invalid_depth.append(l)
            else:
                invalid_rgb.append(l)

    with open("invalid_rgb_nyu.txt", 'w+') as f:
        for l in invalid_rgb:
            f.write(l)
    with open("invalid_depth_nyu.txt", 'w+') as f:
        for l in invalid_depth:
            f.write(l)
