import argparse
import cupy as cp
import numpy as np
from tqdm import tqdm
import mytorch 
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch.utils.data import DataLoader

# Use PyTorch MNIST Class for Simplicity ###
from torchvision.datasets import MNIST

def main(args):

    ### Prep Model ###
    class MyTorchMNIST(nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(784, 512)
            self.drop1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(512, 256)
            self.drop2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(256, 128)
            self.drop3 = nn.Dropout(0.1)
            self.fc4 = nn.Linear(128, 10)

            self.activation = nn.ReLU()

        def forward(self, x):

            x = self.drop1(self.activation(self.fc1(x)))
            x = self.drop2(self.activation(self.fc2(x)))
            x = self.drop3(self.activation(self.fc3(x)))
            x = self.fc4(x)

            return x
        
    model = MyTorchMNIST()
    model = model.to("cuda")
    print(model)

    ### Prep Dataset ###
    train = MNIST("../../data", train=True, download=True)
    test = MNIST("../../data", train=False, download=True)

    def collate_fn(batch):

        ### Prep and Scale Images ###
        images = cp.stack([cp.array(i[0]).reshape(28*28)for i in batch]) / 255

        ### One Hot Encode Label (MNIST only has 10 classes) ###
        labels = [i[1] for i in batch]

        images = mytorch.Tensor(images).astype(cp.float32)
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
            
            ### Set on correct device ###
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
            
            ### Set on correct device ###
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

    parser = argparse.ArgumentParser(description="Train a simple linear classifier using MyTorch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    main(args)