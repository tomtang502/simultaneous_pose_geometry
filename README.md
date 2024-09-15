# Simultaneous Estimation

This  repository is the official implementation of the paper [Simultaneous Estimation of Geometry and Pose of Held Objects
via 3D Foundation Models](https://arxiv.org/pdf/2407.10331). 
![Visualization](/figures/segpo_method.png)

## Installation

It's recommended to use a package manager like [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to create a package environment to install and manage required dependencies, and the following installation guide assume a conda base environment is already initiated.

If unzip is not installed via apt, use apt to install unzip before proceed.
```bash
https://github.com/tomtang502/simultaneous_pose_geometry.git
cd simultaneous_pose_geometry

conda create -n segpo python=3.11 cmake=3.14.0
conda activate segpo
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
sudo chmod +x installation.sh 
./installation.sh
```

## Usage

To download some sample data and the weight for the foundation model () (some of them are processed by tiny-sam for segmentation visualization).
```bash
sudo chmod +x download_data.sh
./download_data.sh
```
[demo.ipynb](demo.ipynb) walk through how to use the process to simultaneously estimate geometry and pose of object hold by a robotic arm, it assume an existing directory containing images of the robot gripper and the object.

[Optional] Set up z1 arm for images taking and operations
The Unitree Z1 Robotics Arm was used for experiments, which comes with z1_controller, z1_ros, and z1_sdk (only z1_controller and z1_sdk are used for taking images and operations). Following their [official documentation](https://dev-z1.unitree.com/) to set up z1 arm for experiments, and [arm_motion](arm_motion) contains the code we used to generate images of robotic gripper and tool.

[MISC]
[run_colmap.py](run_colmap.py) uses colmap to reconstruct the tool and estimate poses [here](https://colmap.github.io/).

## Attribution

This repository includes a module licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

Module: Dust3r

Original Authors: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jérôme Revaud

Source: https://github.com/naver/dust3r

License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
