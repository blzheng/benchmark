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
        self.relu166 = ReLU(inplace=True)
        self.conv2d166 = Conv2d(1376, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x588):
        x589=self.relu166(x588)
        x590=self.conv2d166(x589)
        return x590

m = M().eval()
x588 = torch.randn(torch.Size([1, 1376, 7, 7]))
start = time.time()
output = m(x588)
end = time.time()
print(end-start)
