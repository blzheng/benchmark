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
        self.relu64 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x228, x220):
        x229=operator.add(x228, x220)
        x230=self.relu64(x229)
        x231=self.conv2d70(x230)
        return x231

m = M().eval()
x228 = torch.randn(torch.Size([1, 1024, 14, 14]))
x220 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x228, x220)
end = time.time()
print(end-start)
