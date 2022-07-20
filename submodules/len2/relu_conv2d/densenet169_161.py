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
        self.relu162 = ReLU(inplace=True)
        self.conv2d162 = Conv2d(1568, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x574):
        x575=self.relu162(x574)
        x576=self.conv2d162(x575)
        return x576

m = M().eval()
x574 = torch.randn(torch.Size([1, 1568, 7, 7]))
start = time.time()
output = m(x574)
end = time.time()
print(end-start)
