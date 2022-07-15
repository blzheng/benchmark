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
        self.conv2d252 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x754):
        x755=self.conv2d252(x754)
        return x755

m = M().eval()
x754 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x754)
end = time.time()
print(end-start)
