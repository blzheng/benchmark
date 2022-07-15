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
        self.conv2d290 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x934):
        x935=self.conv2d290(x934)
        return x935

m = M().eval()
x934 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x934)
end = time.time()
print(end-start)
