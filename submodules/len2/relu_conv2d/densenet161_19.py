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
        self.relu20 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x74):
        x75=self.relu20(x74)
        x76=self.conv2d20(x75)
        return x76

m = M().eval()
x74 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x74)
end = time.time()
print(end-start)
