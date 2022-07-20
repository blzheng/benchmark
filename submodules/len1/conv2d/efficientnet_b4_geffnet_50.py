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
        self.conv2d50 = Conv2d(336, 336, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=336, bias=False)

    def forward(self, x149):
        x150=self.conv2d50(x149)
        return x150

m = M().eval()
x149 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x149)
end = time.time()
print(end-start)