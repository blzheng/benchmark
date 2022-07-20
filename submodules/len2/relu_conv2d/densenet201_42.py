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
        self.relu43 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x156):
        x157=self.relu43(x156)
        x158=self.conv2d43(x157)
        return x158

m = M().eval()
x156 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
