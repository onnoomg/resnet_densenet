from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

PATH = './densenet.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('linear1', nn.Linear(num_input, num_input))
        self.add_module('relu1', nn.ReLU())
        self.add_module('norm1', nn.BatchNorm1d(num_input))
        self.add_module('linear2', nn.Linear(num_input, num_input))
        self.add_module('relu2', nn.ReLU())
        self.add_module('norm2', nn.BatchNorm1d(num_input))
        self.drop_rate = drop_rate

    def forward(self, x):
        out = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input, drop_rate, growth_rate, bn_size):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(2 ** i * num_input, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input, num_output):
        super(_Transition, self).__init__()
        self.add_module('linear', nn.Linear(num_input, num_output))
        self.add_module('norm', nn.BatchNorm1d(num_output))
        self.add_module('relu', nn.ReLU())
        # self.add_module('conv', nn.Conv1d(num_input, num_output, kernel_size=1, stride=1, bias=False))


class DenseNet(nn.Module):
    def __init__(self, block_layers, num_init_input, drop_rate, num_classes, growth_rate, bn_size):
        super(DenseNet, self).__init__()

        self.input = nn.Sequential(OrderedDict([
            # ('conv0', nn.Conv1d(1, num_init_input, kernel_size=7, stride=2, padding=3, bias=False)),
            ('linear0', nn.Linear(num_init_input, num_init_input)),
            ('norm0', nn.BatchNorm1d(num_init_input)),
            ('relu0', nn.ReLU()),
            # ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        ]))

        num_input = num_init_input
        for i, num_layer in enumerate(block_layers):
            blk = _DenseBlock(num_layers=num_layer, num_input=num_input,
                              drop_rate=drop_rate, growth_rate=growth_rate, bn_size=bn_size)
            self.input.add_module('denseblk%d' % (i + 1), blk)
            num_input = num_layer ** 2 * num_input
            if i != len(block_layers) - 1:
                trans_blk = _Transition(num_input=num_input, num_output=num_input // 8)
                # trans_blk = _Transition(num_input=num_input, num_output=num_input // 2)
                self.input.add_module('transitionblk%d' % (i + 1), trans_blk)
                # num_input = num_input // 2
                num_input = num_input // 8

        self.input.add_module('norm5', nn.BatchNorm1d(num_input))

        self.clf = nn.Linear(num_input, num_classes)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        input = self.input(x)
        out = F.relu(input)
        out = self.clf(out)
        return out


model = DenseNet(num_init_input=77, block_layers=(4, 4, 4, 4), drop_rate=0,
                 num_classes=18, growth_rate=32, bn_size=4)
model.load_state_dict(torch.load(PATH))
# print(model.state_dict())
model.eval()


def process_frame(file):
    feature = []
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:

        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:

            image = file
            drawimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert the BGR image to RGB before processing.
            results1 = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Print handedness and draw hand landmarks on the image.
            # 识别不到手的直接跳过
            if not results.multi_hand_landmarks:
                return False
            # 识别不到身体部位的也直接跳过
            if not results1.pose_landmarks:
                return False

            # 鼻子
            if results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.8:
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x, 4) * 1000)
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y, 4) * 1000)

            else:
                feature.append(0)
                feature.append(0)

            # 右眼睛
            if results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].visibility > 0.8:
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x, 4) * 1000)
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y, 4) * 1000)
            else:
                feature.append(0)
                feature.append(0)

            # 左眼睛
            if results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].visibility > 0.8:
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x, 4) * 1000)
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y, 4) * 1000)

            else:
                feature.append(0)
                feature.append(0)

            # 嘴巴左侧
            if results1.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].visibility > 0.8:
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x, 4) * 1000)
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y, 4) * 1000)
            else:
                feature.append(0)
                feature.append(0)

            # 嘴巴右侧
            if results1.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].visibility > 0.8:
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x, 4) * 1000)
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y, 4) * 1000)

            else:
                feature.append(0)
                feature.append(0)

            # 左手腕
            if results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.8:
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, 4) * 1000)
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y, 4) * 1000)
            else:
                feature.append(0)
                feature.append(0)
            # 右手腕
            if results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.8:
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, 4) * 1000)
                feature.append(round(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y, 4) * 1000)
            else:
                feature.append(0)
                feature.append(0)

        # Read an image, flip it around y-axis for correct handedness output (see
        # above).

        for i in range(len(results.multi_hand_landmarks)):
            for j in range(21):
                # if j == 4  or  j == 20 or j==12 or j ==16:
                #     feature.append(round(results.multi_hand_landmarks[i].landmark[j].x, 4) * 10000)
                #     feature.append(round(results.multi_hand_landmarks[i].landmark[j].y, 4) * 10000)
                #     feature.append(round(results.multi_hand_landmarks[i].landmark[j].z, 4) * 10000)
                # else:
                feature.append(round(results.multi_hand_landmarks[i].landmark[j].x, 4) * 1000)
                feature.append(round(results.multi_hand_landmarks[i].landmark[j].y, 4) * 1000)
                feature.append(round(results.multi_hand_landmarks[i].landmark[j].z, 4) * 1000)

    if len(feature) != 77:
        return False
    feature = torch.tensor([feature])
    feature = torch.tensor(feature)
    output = model(feature)
    outputindex = output.argmax(dim=-1)
    print(outputindex)
    if outputindex == 0:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'call', (280, 80), font, 2, color, 3)
    elif outputindex == 1:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'dislike', (280, 80), font, 2, color, 3)
    elif outputindex == 2:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'fist', (280, 80), font, 2, color, 3)
    elif outputindex == 3:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'four', (280, 80), font, 2, color, 3)
    elif outputindex == 4:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'like', (280, 80), font, 2, color, 3)
    elif outputindex == 5:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'mute',(280, 80), font, 2, color, 3)
    elif outputindex == 6:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'ok', (280, 80), font, 2, color, 3)
    elif outputindex == 7:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'one', (280, 80), font, 2, color, 3)
    elif outputindex == 8:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'palm', (280, 80), font, 2, color, 3)
    elif outputindex == 9:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'peace', (280, 80), font, 2, color, 3)
    elif outputindex == 10:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'peace_inverted', (280, 80), font, 2, color, 3)
    elif outputindex == 11:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'rock',(280, 80), font, 2, color, 3)
    elif outputindex == 12:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'stop', (280, 80), font, 2, color, 3)
    elif outputindex == 13:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'stop_inverted', (280, 80), font, 2, color, 3)
    elif outputindex == 14:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'three', (280, 80), font, 2, color, 3)
    elif outputindex == 15:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'three2', (280, 80), font, 2, color, 3)
    elif outputindex == 16:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'two_up', (280, 80), font, 2, color, 3)
    elif outputindex == 17:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'two_up_inverted', (280, 80), font, 2, color, 3)

    drawimg = cv2.cvtColor(drawimg, cv2.COLOR_RGB2BGR)
    return drawimg


# 获取摄像头
import cv2

cap = cv2.VideoCapture(0)
cap.open(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print('Error')
        break
    frame = process_frame(frame)
    if type(frame) != bool:
        cv2.imshow('my_wind', frame)
    if cv2.waitKey(1) in [ord('q'), 27]:
        break
cap.release()
cv2.destroyAllWindows()
