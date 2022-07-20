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
        self.relu48 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x173):
        x174=self.relu48(x173)
        x175=self.conv2d48(x174)
        return x175

m = M().eval()
x173 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x173)
end = time.time()
print(end-start)
