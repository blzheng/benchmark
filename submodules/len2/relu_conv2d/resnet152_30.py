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
        self.relu31 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x110):
        x111=self.relu31(x110)
        x112=self.conv2d34(x111)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
