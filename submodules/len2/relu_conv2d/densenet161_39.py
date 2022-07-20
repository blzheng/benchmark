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
        self.relu40 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x145):
        x146=self.relu40(x145)
        x147=self.conv2d40(x146)
        return x147

m = M().eval()
x145 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
