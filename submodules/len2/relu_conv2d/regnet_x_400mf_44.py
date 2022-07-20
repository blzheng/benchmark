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
        self.relu44 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x156):
        x157=self.relu44(x156)
        x158=self.conv2d49(x157)
        return x158

m = M().eval()
x156 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
