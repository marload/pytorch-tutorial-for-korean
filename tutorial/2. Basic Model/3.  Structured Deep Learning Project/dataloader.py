'''
dataloader.py는 주로 Image Preprocessing 코드와 dataloader를 구현합니다.
Image Preprocessing을 아직 배우지 않으셨지만 다음 Basic CNN Classifier를 보시면 이해가 가실겁니다.
'''

import torch
import torchvision
import torchvision.transforms as T
from config import get_config

config = get_config()

train_T = T.Compose([
    T.ToTensor()
])

test_T = T.Compose([
    T.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root=config.dataroot,
                                           train=True,
                                           transform=train_T,
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root=config.dataroot,
                                          train=False,
                                          transform=test_T,
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           num_workers=config.num_workers)
test_loader= torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=config.num_workers)

def get_loader():
    return train_loader, test_loader
