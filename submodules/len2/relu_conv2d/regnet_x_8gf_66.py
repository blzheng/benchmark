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
        self.relu66 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(720, 1920, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x228):
        x229=self.relu66(x228)
        x230=self.conv2d70(x229)
        return x230

m = M().eval()
x228 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x228)
end = time.time()
print(end-start)
