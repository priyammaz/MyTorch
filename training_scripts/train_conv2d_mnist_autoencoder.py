import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch.utils.data import DataLoader

from torchvision.datasets import MNIST

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
batch_size = 64
learning_rate = 0.001
epochs = 10

train_dataset = MNIST(root='../../data', train=True, download=True)

def collate_fn(batch):

    ### Prep and Scale Images ###
    images = np.stack([np.array(i[0]).reshape(1,28,28)for i in batch]) / 255

    ### One Hot Encode Label (MNIST only has 10 classes) ###
    labels = [i[1] for i in batch]
    
    images = mytorch.Tensor(images, dtype=mytorch.float32)
    labels = mytorch.Tensor(labels, dtype=mytorch.int64)

    return images, labels

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# Initialize model, optimizer, and loss
model = Autoencoder().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to("cuda")

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    with mytorch.no_grad():
        sample_imgs, _ = next(iter(train_loader))  # get one batch
        sample_imgs = sample_imgs.to("cuda")
        recons = model(sample_imgs)

    # Convert to numpy for plotting
    sample_imgs = sample_imgs.numpy()
    recons = recons.numpy() 
    # Show first 6 reconstructions
    n = 6
    plt.figure(figsize=(12, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(sample_imgs[i, 0], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        # Reconstructed
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(recons[i, 0], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Reconstructed")
    plt.show()
