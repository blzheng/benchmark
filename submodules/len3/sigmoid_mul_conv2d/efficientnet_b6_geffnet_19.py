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
        self.conv2d97 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x288, x284):
        x289=x288.sigmoid()
        x290=operator.mul(x284, x289)
        x291=self.conv2d97(x290)
        return x291

m = M().eval()
x288 = torch.randn(torch.Size([1, 864, 1, 1]))
x284 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x288, x284)
end = time.time()
print(end-start)
