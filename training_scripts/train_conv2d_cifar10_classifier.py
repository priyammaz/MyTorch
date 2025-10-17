import argparse
import cupy as cp
import numpy as np
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch.utils.data import DataLoader

from torchvision.datasets import CIFAR10

def main(args):

    ### Prep Model ###
    class MyTorchCIFAR10(nn.Module):

        def __init__(self):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                                   stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.drop1 = nn.Dropout(0.1)

            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, 
                                   stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.drop2 = nn.Dropout(0.1)
            
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                   stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.drop3 = nn.Dropout(0.1)
            
            self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                   stride=2, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.drop4 = nn.Dropout(0.1)
            
            self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                   stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
            self.drop5 = nn.Dropout(0.1)

            self.activation = nn.ReLU()
            self.proj = nn.Linear(256*2*2, 10)

        def forward(self, x):
            
            x = self.drop1(self.activation(self.bn1(self.conv1(x))))
            x = self.drop1(self.activation(self.bn2(self.conv2(x))))
            x = self.drop1(self.activation(self.bn3(self.conv3(x))))
            x = self.drop1(self.activation(self.bn4(self.conv4(x))))
            x = self.drop1(self.activation(self.bn5(self.conv5(x))))
     
            x = x.reshape(-1, 256*2*2)
            x = self.proj(x)
            return x
        
    model = MyTorchCIFAR10()
    model = model.to("cuda")

    ### Prep Dataset ###
    train = CIFAR10("../../data", train=True, download=True)
    test = CIFAR10("../../data", train=False, download=True)

    def collate_fn(batch):

        ### Prep and Scale Images ###
        images = np.stack([np.array(i[0]).transpose(2,0,1)for i in batch]) / 255

        ### One Hot Encode Label (MNIST only has 10 classes) ###
        labels = [i[1] for i in batch]

        images = mytorch.Tensor(images).astype(mytorch.float32)
        labels = mytorch.Tensor(labels)

        return images, labels

    trainloader = DataLoader(train, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)
    testloader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

    ### Prep Optimizer ###
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ### Prep Loss Function ###
    loss_fn = nn.CrossEntropyLoss()

    ### Train Model for 10 Epochs ###
    for epoch in range(args.epochs):

        print(f"Training Epoch {epoch}")

        train_loss, train_acc = [], []
        eval_loss, eval_acc = [], []

        model.train()
        for images, labels in tqdm(trainloader):
            
            images, labels = images.to("cuda"), labels.to("cuda")

            ### Pass Through Model ###
            pred = model(images)    

            ### Compute Loss ###
            loss = loss_fn(pred, labels)
     
            ### Compute Accuracy ###
            predicted = pred.argmax(dim=-1)
            accuracy = (predicted == labels).sum() / len(predicted)

            ### Log Results ###
            train_loss.append(loss.item())
            train_acc.append(accuracy.item())
            
            ### Update Model ###
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        model.eval()
        for images, labels in tqdm(testloader):
            
            images, labels = images.to("cuda"), labels.to("cuda")
            
            with mytorch.no_grad():
                ### Pass Through Model ###
                pred = model(images)
                
            ### Compute Loss ###
            loss = loss_fn(pred, labels)

            ### Compute Accuracy ###
            predicted = pred.argmax(dim=-1)
            accuracy = (predicted == labels).sum() / len(predicted)

            eval_loss.append(loss.item())
            eval_acc.append(accuracy.item())

        print(f"Training Loss: {np.mean(train_loss)}")
        print(f"Eval Loss: {np.mean(eval_loss)}")
        print(f"Training Acc: {np.mean(train_acc)}")
        print(f"Eval Acc: {np.mean(eval_acc)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a simple conv classifier using MyTorch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    main(args)