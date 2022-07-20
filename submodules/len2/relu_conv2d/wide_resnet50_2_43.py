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
        self.conv2d49 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x157):
        x158=self.relu43(x157)
        x159=self.conv2d49(x158)
        return x159

m = M().eval()
x157 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x157)
end = time.time()
print(end-start)
