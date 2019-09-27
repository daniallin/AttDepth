import os
import json


root_path = '/Users/doctor_lin/DeeCamp/monoDepthEstimation/hacker'

path1 = os.listdir(root_path)

img_path = list()
for p in path1:
    if p == '.DS_Store':
        continue
    p1 = os.path.join(root_path, p)
    path2 = os.listdir(p1)
    for n in path2:
        file_path = os.path.join(p1, n)
        rgb_depth = dict()
        if "Color" in n:
                rgb_depth['rgb_path'] = file_path
                img_path.append(rgb_depth)

with open("/Users/doctor_lin/DeeCamp/monoDepthEstimation/dataset/hitachi/test.json", "a") as f:
    json.dump(img_path, f)







