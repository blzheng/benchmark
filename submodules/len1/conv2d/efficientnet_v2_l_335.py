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
        self.conv2d335 = Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x1076):
        x1077=self.conv2d335(x1076)
        return x1077

m = M().eval()
x1076 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x1076)
end = time.time()
print(end-start)
