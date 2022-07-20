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
        self.sigmoid21 = Sigmoid()
        self.conv2d128 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x405, x401):
        x406=self.sigmoid21(x405)
        x407=operator.mul(x406, x401)
        x408=self.conv2d128(x407)
        return x408

m = M().eval()
x405 = torch.randn(torch.Size([1, 1536, 1, 1]))
x401 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x405, x401)
end = time.time()
print(end-start)
