Monocular depth estimation for Champion.

# Installation

- Install the required packages in requirements.txt
- Download code from github `https://github.com/daniallin/AttDepth.git`




TODO：

-[x] dataload/kitti_dataset.py


# Preparation for Training:

### NYU Depvh V2

Download NYUv2 dataset. NYUDV2 The details of datasets can be found [here](https://cloudstor.aarnet.edu.au/plus/s/G2ckXCJX3pvrzRU). 

```
$ cd ~/workspace/dataset/nyu_depth_v2
$ mkdir nyuv2_zip && cd nyuv2_zip
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip
$ unzip nyu_depth_v2_raw.zip
```

Create the json files for train and evaluation. The json file named as `train_raw_annotations.json` and `val_raw_annotations.json`. The annotations are like follows.

```
[
 {'depth_path': '<paht>/<image_name>', 'rgb_path': '<path>/<image_name>' }
 ...
]
```


### KITTI

You can also train BTS with KITTI dataset by following procedures. First, download the ground truth depthmaps from KITTI. Then, download and unzip the raw dataset.


Create the json files for train and evaluation. The json file named as `train_raw_annotations.json` and `val_raw_annotations.json`. The annotations are like follows.

```
[
 {'depth_path': '<paht>/<image_name>', 'rgb_path': '<path>/<image_name>' }
 ...
]
```

**Notes**: Change parameters in utils/params.py, include `dataset`,`crop_size`, `dataset_path`



