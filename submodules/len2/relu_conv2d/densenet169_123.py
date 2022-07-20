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
        self.relu124 = ReLU(inplace=True)
        self.conv2d124 = Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x441):
        x442=self.relu124(x441)
        x443=self.conv2d124(x442)
        return x443

m = M().eval()
x441 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x441)
end = time.time()
print(end-start)
