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
        self.relu174 = ReLU(inplace=True)
        self.conv2d174 = Conv2d(1504, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x616):
        x617=self.relu174(x616)
        x618=self.conv2d174(x617)
        return x618

m = M().eval()
x616 = torch.randn(torch.Size([1, 1504, 7, 7]))
start = time.time()
output = m(x616)
end = time.time()
print(end-start)
