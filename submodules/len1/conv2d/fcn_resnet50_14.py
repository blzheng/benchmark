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
        self.conv2d14 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x38):
        x47=self.conv2d14(x38)
        return x47

m = M().eval()
x38 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x38)
end = time.time()
print(end-start)
