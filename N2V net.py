from torch.utils.data import Dataset
from Diffsim import DiffractionPatternSimulator
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class DiffractionPatternDataset(Dataset):
    def __init__(self, count, image_size=256, transform=None):
        """
        :param count: Total number of images to generate.
        :param image_size: Size of the images.
        :param transform: PyTorch transforms to apply to the generated images.
        """
        self.count = count
        self.image_size = image_size
        self.transform = transform
        self.simulator = DiffractionPatternSimulator(image_size=image_size)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        """
        Generates a new diffraction pattern for each call.
        """
        # Reset image in simulator for each generation
        self.simulator.image = np.zeros((self.image_size, self.image_size))

        # Generate a new pattern
        self.simulator.generate_pattern()
        self.simulator.add_gaussian_background()
        self.simulator.add_gaussian_noise(mean=0, std=0.3)  # Customize noise here as needed

        image = self.simulator.image

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        # Convert image to PyTorch tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension

        return image


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def apply_mask(self, inputs, mask_value=0):
        """
        Apply a random mask to the center pixel in each patch.
        """
        N, C, H, W = inputs.shape
        mask = torch.ones_like(inputs)
        for n in range(N):
            for h in range(1, H - 1):
                for w in range(1, W - 1):
                    if np.random.rand() <= 0.5:  # Random masking with a probability of 0.5
                        mask[n, :, h, w] = mask_value
        return inputs * mask

    def training_step(self, batch, batch_idx):
        inputs = batch
        masked_inputs = self.apply_mask(inputs)
        outputs = self(masked_inputs)
        loss = F.mse_loss(outputs, inputs)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


train_dataset = DiffractionPatternDataset(count=1000)  # Adjust count as needed
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model Initialization
model = UNet(n_channels=1, n_classes=1)  # Adjust for your dataset specifics

# Training Configuration
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Adjust based on your validation metric
    dirpath='model_checkpoints',
    filename='noise2void-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

trainer = Trainer(
    accelerator='auto',  # Utilize GPU if available
    max_epochs=50,  # Adjust based on your needs
    callbacks=[checkpoint_callback]
)

# Training the Model
trainer.fit(model, train_loader)
