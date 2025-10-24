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

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Downsample path if needed (for matching dimensions)
            self.downsample = downsample

        def forward(self, x):
            identity = x  # shortcut connection
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            if self.downsample is not None:
                identity = self.downsample(x)  # match dimensions
            
            out = out + identity  # residual connection
            out = self.relu(out)
            
            return out

    class MyTorchCIFAR10(nn.Module):
        def __init__(self):
            super(MyTorchCIFAR10, self).__init__()
            
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            
            # Adding residual blocks
            self.layer1 = self._make_layer(16, 32, stride=2)
            self.layer2 = self._make_layer(32, 64, stride=2)
            self.layer3 = self._make_layer(64, 128, stride=2)
            self.layer4 = self._make_layer(128, 128, stride=1)
            
            self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
            self.fc = nn.Linear(512, 10)  # Final fully connected layer for 10 classes
        
        def _make_layer(self, in_channels, out_channels, stride):
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
            
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
            layers.append(ResidualBlock(out_channels, out_channels))  # Second block
            
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))  # Initial convolution and batch norm
            x = self.layer1(x)  # First residual block layer
            x = self.layer2(x)  # Second residual block layer
            x = self.layer3(x)  # Third residual block layer
            x = self.layer4(x)  # Fourth residual block layer
            x = x.reshape(-1, 128*2*2)
            x = self.fc(x)  # Final classification layer
            
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