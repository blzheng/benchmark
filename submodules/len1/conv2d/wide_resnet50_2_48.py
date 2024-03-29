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
        self.conv2d48 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x155):
        x156=self.conv2d48(x155)
        return x156

m = M().eval()
x155 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x155)
end = time.time()
print(end-start)
