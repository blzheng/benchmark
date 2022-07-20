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
        self.conv2d23 = Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x90):
        x91=torch.nn.functional.relu(x90,inplace=True)
        x92=self.conv2d23(x91)
        return x92

m = M().eval()
x90 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x90)
end = time.time()
print(end-start)
