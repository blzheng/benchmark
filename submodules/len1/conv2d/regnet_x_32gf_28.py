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
        self.conv2d28 = Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)

    def forward(self, x90):
        x91=self.conv2d28(x90)
        return x91

m = M().eval()
x90 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x90)
end = time.time()
print(end-start)
