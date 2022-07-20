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
        self.conv2d29 = Conv2d(208, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x73, x87):
        x88=operator.add(x73, x87)
        x89=self.relu20(x88)
        x90=self.conv2d29(x89)
        return x90

m = M().eval()
x73 = torch.randn(torch.Size([1, 208, 14, 14]))
x87 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x73, x87)
end = time.time()
print(end-start)
