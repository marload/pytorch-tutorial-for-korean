'''
config.py는 프로젝트에 필요한 여러 HyperParameter를 담는 파일입니다.
argparse라는 패키지를 사용해 프로그램을 실행할 때 하이퍼파라미터들을 입력받습니다.
'''

import argparse

parser = argparse.ArgumentParser()

# Dataloader Config
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=1)

# Model Config
parser.add_argument('--image_size', type=int, default=28)
parser.add_argument('--nb_classes', type=int, default=10)

# Test Config
parser.add_argument('--save_model_dir', type=str, default='./models')
parser.add_argument('--ckpt_name', type=str, default='model.ckpt')

# Step Size
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)

def get_config():
    config = parser.parse_args()
    return config