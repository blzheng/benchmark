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
        self.conv2d21 = Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x62):
        x63=self.conv2d21(x62)
        return x63

m = M().eval()
x62 = torch.randn(torch.Size([1, 120, 1, 1]))
start = time.time()
output = m(x62)
end = time.time()
print(end-start)
