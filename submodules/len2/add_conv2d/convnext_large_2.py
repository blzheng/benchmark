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
        self.conv2d6 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)

    def forward(self, x55, x45):
        x56=operator.add(x55, x45)
        x58=self.conv2d6(x56)
        return x58

m = M().eval()
x55 = torch.randn(torch.Size([1, 384, 28, 28]))
x45 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x55, x45)
end = time.time()
print(end-start)