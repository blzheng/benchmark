import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.relu87 = ReLU()
        self.conv2d112 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()

    def forward(self, x353):
        x354=self.relu87(x353)
        x355=self.conv2d112(x354)
        x356=self.sigmoid21(x355)
        return x356

m = M().eval()
x353 = torch.randn(torch.Size([1, 84, 1, 1]))
start = time.time()
output = m(x353)
end = time.time()
print(end-start)
