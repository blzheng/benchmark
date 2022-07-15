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
        self.conv2d110 = Conv2d(1920, 1920, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1920, bias=False)

    def forward(self, x338):
        x339=self.conv2d110(x338)
        return x339

m = M().eval()
x338 = torch.randn(torch.Size([1, 1920, 7, 7]))
start = time.time()
output = m(x338)
end = time.time()
print(end-start)
