- https://www.anaconda.com/download/success
- https://github.com/akanametov/yolov8-face
- https://github.com/ultralytics/ultralytics
- https://github.com/ultralytics/ultralytics
- https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
- https://pytorch.org/get-started/previous-versions/

```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
Add Miniconda3 to my PATH environment variable

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

CUDA 10.2
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

```
conda create -n yolov8 python=3.8
conda activate yolov8
pip install -e .
pip list
```

```
pip install opencv-python 
pip install labelimg 
pip install mediapipe
pip install facenet-pytorch
 
pip install jupyterlab

yolo predict model=yolov8n.pt source=0 show=True save=True 
yolo predict model=yolov8n-face.pt source=0 show=True save=True 
```

```
source="/assets/bus.jpg" 检测图片
source="screen"  检测电脑桌面
source=0  检测摄像头

conf=0.5 值越小 检测框越多
iou=0.7  值越小 检测框越少
```

## <div align="center">License</div>

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/licenses/) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://ultralytics.com/license).
