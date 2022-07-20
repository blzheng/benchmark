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
        self.relu83 = ReLU()
        self.conv2d108 = Conv2d(144, 1512, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x339):
        x340=self.relu83(x339)
        x341=self.conv2d108(x340)
        return x341

m = M().eval()
x339 = torch.randn(torch.Size([1, 144, 1, 1]))
start = time.time()
output = m(x339)
end = time.time()
print(end-start)
