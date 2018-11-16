'''
model.py는 딥러닝 모델을 담는 파일입니다. 주로 model 클래스를 구현해 놓습니다.
'''

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(config.image_size**2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, config.nb_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
