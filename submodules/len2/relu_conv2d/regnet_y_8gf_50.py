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
        self.relu67 = ReLU()
        self.conv2d88 = Conv2d(224, 2016, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x275):
        x276=self.relu67(x275)
        x277=self.conv2d88(x276)
        return x277

m = M().eval()
x275 = torch.randn(torch.Size([1, 224, 1, 1]))
start = time.time()
output = m(x275)
end = time.time()
print(end-start)
