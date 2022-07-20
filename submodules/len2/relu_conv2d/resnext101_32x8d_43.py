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
        self.conv2d48 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x155):
        x156=self.relu43(x155)
        x157=self.conv2d48(x156)
        return x157

m = M().eval()
x155 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x155)
end = time.time()
print(end-start)
