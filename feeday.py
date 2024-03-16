import cv2
import os
import numpy as np
import math
import shutil
import argparse
import glob
import torch
import glob
import mediapipe as mp
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# 代码来自以下项目通过GPT4修改增加的功能

# https://github.com/akanametov/yolov8-face
# https://github.com/ultralytics/ultralytics

# 下面是整理所需要的安装的链接
# https://github.com/ultralytics/ultralytics
# https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
# https://pytorch.org/get-started/previous-versions/

# https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
# Add Miniconda3 to my PATH environment variable

# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# CUDA 10.2
# pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
# CUDA 11.6
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116


# conda create -n yolov8 python=3.8
# conda activate yolov8
# pip install -e .
# pip list

# pip install opencv-python 
# pip install labelimg 
# pip install mediapipe
# pip install facenet-pytorch
 
# pip install jupyterlab

# yolo predict model=yolov8n.pt source=0 show=True save=True 
# yolo predict model=yolov8n-face.pt source=0 show=True save=True 

# source="/assets/bus.jpg" 检测图片
# source="screen"  检测电脑桌面
# source=0  检测摄像头

# conf=0.5 值越小 检测框越多
# iou=0.7  值越小 检测框越少

# 先执行这个如果可以用 # 注释掉然后执行以下代码
######################################################################
yolo = YOLO("yolov8n-face.pt")                                      #
result = yolo(source=0, show=True, conf=0.4, save=True)             #
######################################################################


# https://docs.ultralytics.com/
# https://github.com/ultralytics/ultralytics/issues

# 图片不支持中文


# 处理后同人图片路径：F:\save\face
# 处理后单人图片路径：F:\save\face1
# 处理后多人图片路径：F:\\save\face2
# 处理后其他图片路径：F:\save\other

# 默认图片路径：\images
# 临时图片路径：F:\save\yolov8
# 默认图片路径：F:\feeday\runs\detect
# 默认模型路径：F:\feeday\yolov8n.pt
# 人脸模型路径：F:\feeday\yolov8n-face.pt
# 五官模型路径：F:\feeday\yolov8n-face.onnx

# 单人脸和多人脸图片分类
class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5, single_person_dir='single_person', multiple_people_dir='multiple_people'):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

        single_person_dir = r"F:\save\face1" 
        multiple_people_dir = r"F:\save\face2" 
        other_dir=r"F:\save\yolov8" # 创建临时文件夹

        self.single_person_dir = single_person_dir
        self.multiple_people_dir = multiple_people_dir
        self.other_dir = other_dir
        
        if not os.path.exists(self.other_dir):
            os.makedirs(self.other_dir)
        if not os.path.exists(self.single_person_dir):
            os.makedirs(self.single_person_dir)
        if not os.path.exists(self.multiple_people_dir):
            os.makedirs(self.multiple_people_dir)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0]/newh, srcimg.shape[1]/neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # if isinstance(outputs, tuple):
        #     outputs = list(outputs)
        # if float(cv2.__version__[:3])>=4.7:
        #     outputs = [outputs[2], outputs[0], outputs[1]] ###opencv4.7需要这一步，opencv4.5不需要
        # Perform inference on the image
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
       #indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,self.iou_threshold).flatten()

        # 检查indices是否为空的元组，如果是，则处理为一个空的NumPy数组
        if isinstance(indices, tuple) and len(indices) == 0:
            indices = np.array([])
        else:
            indices = indices.flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
 
    def draw_detections(self, image, boxes, scores, kpts):  #####检测框关####
#       for box, score, kp in zip(boxes, scores, kpts):
#           x, y, w, h = box.astype(int)
#           # Draw rectangle
#           cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
#           cv2.putText(image, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
#           for i in range(5):
#               cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
#               cv2.putText(image, str(i), (int(kp[i * 3]), int(kp[i * 3 + 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        return image
 
 


    def save_detections(self, srcimg, boxes, dstimg_path):
        # 根据检测到的人脸数量，将图像保存到适当的目录
        if len(boxes) == 0:
            # 没有检测到人脸
            save_path = os.path.join(self.other_dir, dstimg_path)
        elif len(boxes) == 1:
            # 检测到一个人脸
            save_path = os.path.join(self.single_person_dir, dstimg_path)
        elif len(boxes) >= 2:
            # 检测到多个人脸
            save_path = os.path.join(self.multiple_people_dir, dstimg_path)
        else:
            # 默认情况，虽然实际上可能永远不会执行到这里
            
            save_path = os.path.join(self.other_dir, dstimg_path)

        cv2.imwrite(save_path, srcimg)
        print(f"Image saved to {save_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLOv8 Face Detection')
    parser.add_argument('--path', type=str, default='./images', help="a directory containing images")
    parser.add_argument('--modelpath', type=str, default='./yolov8n-face.onnx', help="model filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence threshold')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='NMS IoU threshold')
    args = parser.parse_args()

    # Initialize YOLOv8_face detector
    YOLOv8_face_detector = YOLOv8_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)

    if os.path.isdir(args.path):
        # Handle directory containing images
        for img_file in glob.glob(os.path.join(args.path, '*')):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                srcimg = cv2.imread(img_file)
                # Perform detection, drawing, and saving for each image
                boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)
                dstimg = YOLOv8_face_detector.draw_detections(srcimg, boxes, scores, kpts)
                YOLOv8_face_detector.save_detections(dstimg, boxes, os.path.basename(img_file))
    else:
        # Handle single image file
        srcimg = cv2.imread(args.path)
        boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)
        dstimg = YOLOv8_face_detector.draw_detections(srcimg, boxes, scores, kpts)
        YOLOv8_face_detector.save_detections(dstimg, boxes, os.path.basename(args.path))
        cv2.imshow('Image', dstimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 人脸分类后其他的图片分类

def detect_and_save_with_yolo(model_path, input_dir, output_dir, conf=0.4, show=True):
    """
    使用YOLO模型检测输入目录中的图像，并将结果保存到输出目录。

    参数:
    - model_path: YOLO模型的路径。
    - input_dir: 包含要检测图像的输入目录。
    - output_dir: 检测结果的输出目录。
    - conf: 置信度阈值。
    - show: 是否显示检测结果。
    """
    # 初始化YOLO模型
    model = YOLO(model_path)

    # 执行检测
    results = model(source=input_dir, show=False, conf=0.5, save=True)

    # 获取默认保存检测结果的目录
    default_save_dir = './runs/detect'
    latest_exp = sorted(os.listdir(default_save_dir))[-1]
    latest_exp_dir = os.path.join(default_save_dir, latest_exp)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 将检测结果从默认目录复制到指定目录
    for file_name in os.listdir(latest_exp_dir):
        source_file = os.path.join(latest_exp_dir, file_name)
        destination_file = os.path.join(output_dir, file_name)
        shutil.copy2(source_file, destination_file)

    print("图片已分类保存到：F:\\save")
    shutil.rmtree("F:\save\yolov8")


# 调用函数示例
model_path = 'yolov8n.pt'
input_dir = r"F:\save\yolov8"
output_dir = r"F:\save\other"


detect_and_save_with_yolo(model_path, input_dir, output_dir)



# 初始化MTCNN和InceptionResnetV1
mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 定义函数获取图片的embedding
def get_embedding(image_path):
    img = Image.open(image_path)
    img_cropped = mtcnn(img)
    with torch.no_grad():
        embedding = resnet(img_cropped.unsqueeze(0))
    return embedding.squeeze().cpu().numpy()

# 加载所有图片路径
image_dir = r'F:\save\face1'
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

# 计算所有图片的embeddings
embeddings = np.array([get_embedding(path) for path in image_paths])

# 计算相似度矩阵
similarity_matrix = np.dot(embeddings, embeddings.T)
# 归一化相似度值到[0,1]
norms = np.linalg.norm(embeddings, axis=1)
similarity_matrix = similarity_matrix / np.outer(norms, norms)

# 分组逻辑，简单阈值判断，阈值可以根据需要调整
threshold = 0.6
groups = []
used = set()

for i in range(len(similarity_matrix)):
    if i in used:
        continue
    # 找到与图片i相似度高于阈值的所有图片
    similar_indices = np.where(similarity_matrix[i] > threshold)[0]
    if len(similar_indices) > 0:
        groups.append([image_paths[idx] for idx in similar_indices])
        used.update(similar_indices)

# 输出分组信息，并移动到新的文件夹
output_dir_base = r'F:\save\face'

# 暂存需要删除的文件路径
files_to_delete = []

for i, group in enumerate(groups):
    # 检查是否至少有两种不同的图片
    unique_images = set()
    for path in group:
        unique_images.add(os.path.basename(path))
    if len(unique_images) < 2:
        continue  # 如果图片种类少于2，不进行复制操作
    
    group_dir = os.path.join(output_dir_base, str(i))
    os.makedirs(group_dir, exist_ok=True)
    for path in group:
        shutil.copy2(path, os.path.join(group_dir, os.path.basename(path)))
        files_to_delete.append(path)  # 将文件路径添加到删除列表中

# 删除原始文件
for file_path in files_to_delete:
    if os.path.exists(file_path):
        os.remove(file_path)  # 在删除文件之前先检查文件是否存在

# 设置目录路径
output_dir_base = Path(r'F:\save\face')

# 计算并打印结果
dir_count = sum(1 for _ in output_dir_base.iterdir() if _.is_dir())
print(f"已发现 {dir_count} 个人的相似图片已保存到：F:\\save\\face")


