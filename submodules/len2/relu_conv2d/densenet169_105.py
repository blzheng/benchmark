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
        self.relu106 = ReLU(inplace=True)
        self.conv2d106 = Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x378):
        x379=self.relu106(x378)
        x380=self.conv2d106(x379)
        return x380

m = M().eval()
x378 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x378)
end = time.time()
print(end-start)
