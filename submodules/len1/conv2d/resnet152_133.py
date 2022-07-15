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
        self.conv2d133 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x440):
        x441=self.conv2d133(x440)
        return x441

m = M().eval()
x440 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x440)
end = time.time()
print(end-start)
