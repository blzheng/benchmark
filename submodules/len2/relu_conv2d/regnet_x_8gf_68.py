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
        self.relu68 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(1920, 1920, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x236):
        x237=self.relu68(x236)
        x238=self.conv2d73(x237)
        return x238

m = M().eval()
x236 = torch.randn(torch.Size([1, 1920, 7, 7]))
start = time.time()
output = m(x236)
end = time.time()
print(end-start)
