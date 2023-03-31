# SADI - YOLOv5 

This is a fork of the [ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)

## Windows Requirements

- Python 3.6+
- NVIDIA CUDA 11.7
- Anaconda (For Environment Management)
    - Installation Guide here: https://docs.anaconda.com/anaconda/install/windows/
    - Open Anaconda Prompt
      - Look for the anaconda installation directory: 
            ``where conda``
            ``where python``
    - Copy the path of the python.exe file and paste it in the environment variable path
      - Example: ``C:\Users\Jas\anaconda3\``
    - Copy the path of the conda.exe file and paste it in the environment variable path
      - Example: ``C:\Users\Jas\anaconda3\Scripts\``
    - Copy the path of anaconda script and paste it in the environment variable path
      - Example: ``C:\Users\Jas\anaconda3\Library\bin`` 
    - Restart the computer

## Windows Installation

- Clone the repository
```git clone https://github.com/code-jas/SADI-Yolov5-Lite.git```
- Open Anaconda Prompt
- Create a new environment
    ``conda create --name yolov5 python=3.8``
- Activate the environment
    ``conda activate yolov5``
- Install the requirements
    ``pip install -r requirements.txt``
- Install Pytorch and Torch (For GPU support: NVIDIA CUDA 11.7)
    ``pip install torch===1.13.0+cu117 torchvision===0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html``
- Install Pycocotools
    ``conda install -c conda-forge pycocotools``

## Training Dataset

- Download the dataset from here: 
    `` wget -c "https://drive.google.com/uc?id=1fhhMEXFtTTYmdnDdUSNrv5Uhaf6vFLjq&confirm=t&uuid=9cc8ab3a-1c16-4109-93ba-392c3c17efbd" ``
- Rename the zip file to dataset.zip
- Extract the zip file

### Train a model using the downloaded dataset above 

The following scripts are available for training the dataset above:

- Train the model
  ``python train.py --data persondog/data.yaml --cfg models/v5Lite-s.yaml --weights weights/v5lite-s.pt --batch-size 32``

## Scripts (GENERAL USAGE)

The following scripts are available for general usage:

- Train the model
    ``python train.py --img 640 --batch 16 --epochs 3 --weights yolov5s.pt --cache``
- Use Webcam Inference
    ``python detect.py --source 0 --weights <path to weights>``
- Use Video Inference
    ``python detect.py --source <path to video> --weights <path to weights>``
- Use Image Inference
    ``python detect.py --source <path to image> --weights <path to weights>``


## Changes

CCTV - Delayed real-time object detection the more time we run the model the more it delays.
Fix: https://github.com/ultralytics/yolov5/issues/4465#issuecomment-1113038325

datasets.py:362 comment  time.sleep(1 / self.fps[i]) # wait time

Fix the rotation augmentation of the image using roboflow
* first we augment the images we annotate using roboflow including rotation augmentation after the augmentation we notice that the bounding box expanded that's why we decided to not use the rotation augmentation.
