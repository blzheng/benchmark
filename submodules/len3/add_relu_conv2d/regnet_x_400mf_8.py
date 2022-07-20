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
        self.relu27 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x89, x97):
        x98=operator.add(x89, x97)
        x99=self.relu27(x98)
        x100=self.conv2d31(x99)
        return x100

m = M().eval()
x89 = torch.randn(torch.Size([1, 160, 14, 14]))
x97 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x89, x97)
end = time.time()
print(end-start)
