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
        self.relu50 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x176):
        x177=self.relu50(x176)
        x178=self.conv2d55(x177)
        return x178

m = M().eval()
x176 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
