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
        self.conv2d57 = Conv2d(40, 640, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x187, x184):
        x188=self.conv2d57(x187)
        x189=self.sigmoid6(x188)
        x190=operator.mul(x189, x184)
        return x190

m = M().eval()
x187 = torch.randn(torch.Size([1, 40, 1, 1]))
x184 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x187, x184)
end = time.time()
print(end-start)
