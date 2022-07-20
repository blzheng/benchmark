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
        self.relu100 = ReLU(inplace=True)
        self.conv2d100 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x355):
        x356=self.relu100(x355)
        x357=self.conv2d100(x356)
        return x357

m = M().eval()
x355 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x355)
end = time.time()
print(end-start)
