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
        self.relu22 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)

    def forward(self, x109):
        x110=self.relu22(x109)
        x111=self.conv2d34(x110)
        return x111

m = M().eval()
x109 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
