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
        self.conv2d127 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x398, x383):
        x399=operator.add(x398, x383)
        x400=self.conv2d127(x399)
        return x400

m = M().eval()
x398 = torch.randn(torch.Size([1, 160, 14, 14]))
x383 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x398, x383)
end = time.time()
print(end-start)