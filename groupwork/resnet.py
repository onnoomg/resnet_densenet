import torch.utils.data
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, num_hid, bias=True, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.linear1 = nn.Linear(num_hid, num_hid, bias)
        self.Batch = nn.BatchNorm1d(num_hid)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(num_hid, num_hid, bias)
        self.tanh2 = nn.Tanh()

    def forward(self, input):
        l1 = self.linear1(input)
        b1 = self.Batch(l1)
        result = self.tanh2(input + b1)
        return result


class Resnet(nn.Module):
    def __init__(self, num_hid, num_layers, num_output, bias=True, **kwargs):
        super(Resnet, self).__init__(**kwargs)
        self.layers = nn.Sequential()
        self.lastlinear = nn.Linear(num_hid, num_output)
        self.softmax = nn.Softmax(dim=-1)
        for i in range(num_layers):
            self.layers.add_module('layer' + str(i),
                                   ResnetBlock(num_hid, bias))

    def forward(self, input, *args):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.lastlinear(x)
        x = self.softmax(x)
        return x


class DatasetD(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


path = 'feature/'
# 训练集
index = 0
feature = []
label = []
for subdir in os.listdir(path):
    print(subdir)
    if index==8:
        break
    with open(path + subdir, 'r') as f:
        file = f.read()
        file = file.split(']')
        file = file[:-1]
        print(len(file))
        for i in range(len(file)):
            file[i] = file[i].split(',')
            file[i] = file[i][:-1]

            if len(file[i]) == 77:
                for j in range(len(file[i])):
                    # if (j >= 26 and j <= 28) or (j >= 74 and j <= 76) or (j >= 62 and j <= 64) or (j >= 50 and j <= 52) :
                    #     file[i][j] = float(file[i][j]) * 10000
                    # else:
                        file[i][j] = float(file[i][j]) * 1000
                if index == 0:
                    label.append(torch.tensor([1., 0., 0., 0., 0.,0.,0., 0.]))
                elif index == 1:
                    label.append(torch.tensor([0., 1., 0., 0., 0.,0.,0., 0.]))
                elif index == 2:
                    label.append(torch.tensor([0., 0., 1., 0., 0.,0.,0., 0.]))
                elif index == 3:
                    label.append(torch.tensor([0., 0., 0., 1., 0.,0.,0., 0.]))
                elif index == 4:
                    label.append(torch.tensor([0., 0., 0., 0., 1.,0.,0., 0.]))
                elif index == 5:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 1.,0., 0.]))
                elif index == 6:
                    label.append(torch.tensor([0., 0., 0., 0., 0.,0.,1., 0.]))
                elif index == 7:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0.,0., 1.]))

                feature.append(torch.tensor(file[i]))
    index += 1

feature = torch.stack(feature, 0)
label = torch.stack(label, 0)

train_dataset = DatasetD(feature, label)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

net = Resnet(77, 25, index)


def train(net, train_loader, optimizer):
    total = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # zero the gradients
        output = net(data)  # apply network
        loss = F.cross_entropy(output, target)

        loss.backward()  # compute gradients
        optimizer.step()  # update weights
        outputindex = output.argmax(dim=-1)
        predicindex = target.argmax(dim=-1)
        correct += (outputindex == predicindex).float().sum()
        total += target.size()[0]
        # correct += (pred == target).float().sum()
        # total += target.size()[0]

    avgaccuracy = correct / total
    # avgaccuracy=avgaccuracy/batch_num
    return avgaccuracy


optimizer = torch.optim.Adam(net.parameters(), lr=0.002,
                             weight_decay=0.001)

epoch = 0
count = 0
while epoch < 20000 and count < 50:
    epoch = epoch + 1
    avgaccuracy = train(net, train_loader, optimizer)
    print(avgaccuracy)
    if avgaccuracy > 0.91:
        count += 1
    if avgaccuracy > 0.91 and count == 10:
        torch.save(net.state_dict(), 'resnet.pth')
        print('Done')
        break

net.eval()

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


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
    print(feature.shape)
    output = net(feature)
    outputindex = output.argmax(dim=-1)
    print(outputindex)
    if outputindex == 0:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'call', (100, 100), font, 5, color, 3)
    elif outputindex == 1:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'dislike', (100, 100), font, 5, color, 3)
    elif outputindex == 2:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'fist', (100, 100), font, 5, color, 3)
    elif outputindex == 3:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'four', (100, 100), font, 5, color, 3)
    elif outputindex == 4:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'like', (100, 100), font, 5, color, 3)
    elif outputindex == 5:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'mute', (100, 100), font, 5, color, 3)
    elif outputindex == 6:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'ok', (100, 100), font, 5, color, 3)
    elif outputindex == 7:
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (10, 20, 20)
        drawimg = cv2.putText(drawimg, 'one', (100, 100), font, 5, color, 3)

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
