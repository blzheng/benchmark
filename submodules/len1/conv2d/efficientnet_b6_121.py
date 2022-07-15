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
        self.conv2d121 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x377):
        x378=self.conv2d121(x377)
        return x378

m = M().eval()
x377 = torch.randn(torch.Size([1, 50, 1, 1]))
start = time.time()
output = m(x377)
end = time.time()
print(end-start)
