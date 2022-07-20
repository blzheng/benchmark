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
        self.relu72 = ReLU(inplace=True)
        self.conv2d77 = Conv2d(1008, 1008, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x250):
        x251=self.relu72(x250)
        x252=self.conv2d77(x251)
        return x252

m = M().eval()
x250 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x250)
end = time.time()
print(end-start)