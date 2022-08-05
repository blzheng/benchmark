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
        self.conv2d97 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x312):
        x321=self.conv2d97(x312)
        return x321

m = M().eval()
x312 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x312)
end = time.time()
print(end-start)
