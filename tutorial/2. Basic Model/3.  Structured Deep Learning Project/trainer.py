'''
trainer.py는 학습을 위한 클래스를 담은 파일입니다.
'''

import torch
import os

class Trainer:
    def __init__(self, model, config, train_loader, test_loader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_path = os.path.join(config.save_model_dir, config.ckpt_name)

        model.to(self.device)

        if not os.path.exists(config.save_model_dir):
            os.mkdir(config.save_model_dir)

        if os.path.isfile(self.ckpt_path):
            print("{} is exist".format(config.ckpt_name))
            exit(-1)

        self.train(model, config, train_loader, test_loader)

    def train(self, model, config, train_loader, test_loader):
        input_size = config.image_size**2

        cost_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        total_step = len(train_loader)
        for epoch in range(config.num_epochs):
            for idx, (images, labels) in enumerate(train_loader):
                images = images.view(-1, input_size).to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = cost_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not (idx+1) % 100:
                    print("Epoch[{}/{}], Batch[{}/{}], Loss={}"
                          .format(epoch + 1, config.num_epochs,
                                  idx + 1, total_step, loss.item()))

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, input_size).to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Accuracy: {}".format(100 * correct / total))

        self.save_model(model)


    def save_model(self, model):
        while True:
            wantSave = input("Do you want SAVE? (Y/N): ")

            if wantSave == 'Y' or wantSave == 'y':
                wantSave = True
                break
            elif wantSave == 'N' or wantSave == 'n':
                wantSave = False
                break
            else:
                print("Wrong")

        if wantSave:
            torch.save(model.state_dict(), self.ckpt_path)
            print("Model Save PATH=> {}".format(self.ckpt_path))
        else:
            print("Okay")