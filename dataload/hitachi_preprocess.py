import os
import json
import random


root_path = '/home/catalina/depth_estimate/dataset/hacker'

path1 = os.listdir(root_path)

img_path = list()
for p in path1:
    p1 = os.path.join(root_path, p)
    path2 = os.listdir(p1)
    for d in path2:
        p2 = os.path.join(p1, d)
        img_name = os.listdir(p2)
        for n in img_name:
            file_path = os.path.join(p2, n)
            rgb_depth = dict()
            if "Color" in n:
                depth_name = 'Depth_' + n.split('_')[-1].split('.')[0] + '.png'
                if depth_name in img_name:
                    depth_path = os.path.join(p2, depth_name)
                    rgb_depth['rgb_path'] = file_path
                    rgb_depth['depth_path'] = depth_path
                    img_path.append(rgb_depth)

random.shuffle(img_path)
img_num = len(img_path)
with open("/home/catalina/depth_estimate/Champion/dataset/hitachi/train_annotations.json", "a") as f:
    json.dump(img_path[:int(img_num * 0.9)], f)
with open("/home/catalina/depth_estimate/Champion/dataset/hitachi/val_annotations.json", "a") as f:
    json.dump(img_path[int(img_num * 0.9):], f)







