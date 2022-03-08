from torch import nn
import torch.nn.functional as F

# define the CNN model for PyTorch
class ConvNet1D(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))        # [-, 9, 128]
        self.layer2 = nn.Flatten()   # [-, 768]
        self.layer3 = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU())               # [-, 100]
        self.layer4 = nn.Sequential(
            nn.Linear(100, 6),
            nn.Softmax())            # [-, 6]

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
