from model import Model
from trainer import Trainer
from config import get_config
from dataloader import get_loader

def main():
    config = get_config()
    train_loader, test_laoder = get_loader()

    model = Model(config)
    trainer = Trainer(model, config, train_loader, test_laoder)

if __name__ == '__main__':
    main()

