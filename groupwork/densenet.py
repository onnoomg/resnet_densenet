from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os


class DatasetD(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


path = './feature/'
# 训练集
index = 0
feature = []
label = []
for subdir in os.listdir(path):
    print(subdir)

    with open(path + subdir, 'r') as f:
        file = f.read()
        file = file.split(']')
        file = file[:-1]
        for i in range(len(file)):
            file[i] = file[i].split(',')
            file[i] = file[i][:-1]

            if len(file[i]) == 77:
                for j in range(len(file[i])):
                    # if (j >= 23 and j <= 29) or (j >= 68 and j <= 76):
                    # if (j >= 6 and j <= 9):
                    #     file[i][j] = float(file[i][j]) * 10000
                    # else:
                    file[i][j] = float(file[i][j]) * 1000
                if index == 0:
                    label.append(torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 1:
                    label.append(torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 2:
                    label.append(torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 3:
                    label.append(torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 4:
                    label.append(torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 5:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 6:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 7:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 8:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 9:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 10:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))
                elif index == 11:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]))
                elif index == 12:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]))
                elif index == 13:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]))
                elif index == 14:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]))
                elif index == 15:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]))
                elif index == 16:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]))
                elif index == 17:
                    label.append(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]))
                feature.append(torch.tensor(file[i]))
    index += 1

feature = torch.stack(feature, 0)
label = torch.stack(label, 0)

train_dataset = DatasetD(feature, label)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


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
            layer = _DenseLayer(2**i * num_input, drop_rate)
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
        self.softmax = nn.Softmax(dim=-1)
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
        out = self.softmax(out)
        return out
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, train_loader, optimizer):
    total = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # zero the gradients
        # x, y = data.shape
        # data = data.reshape(x, 1, y)
        data,target = data.to(device),target.to(device)
        output = net(data)  # apply network
        loss = F.cross_entropy(output, target)
        loss.backward()  # compute gradients
        optimizer.step()  # update weights
        outputindex = output.argmax(dim=-1)
        predicindex = target.argmax(dim=-1)

        correct += (outputindex == predicindex).float().sum()
        total += target.size()[0]
    avgaccuracy = correct / total

    return avgaccuracy

# 6 12 24 16

model = DenseNet(num_init_input=77, block_layers=(4, 4, 4, 4), drop_rate=0.4,
                 num_classes=18, growth_rate=32, bn_size=4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.004,
                             weight_decay=0.001)

epoch = 0
count = 0
while epoch < 50000 and count < 50:
    epoch += 1
    avgaccuracy = train(model, train_loader, optimizer)
    print(avgaccuracy)
    if avgaccuracy > 0.95:
        count += 1
    if avgaccuracy > 0.95 and count == 20:
        torch.save(model.state_dict(), 'densenet.pth')
        print('Done')
        break

model.eval()





