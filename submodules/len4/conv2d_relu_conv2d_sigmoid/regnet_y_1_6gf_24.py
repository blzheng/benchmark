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
        self.conv2d126 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu99 = ReLU()
        self.conv2d127 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()

    def forward(self, x400):
        x401=self.conv2d126(x400)
        x402=self.relu99(x401)
        x403=self.conv2d127(x402)
        x404=self.sigmoid24(x403)
        return x404

m = M().eval()
x400 = torch.randn(torch.Size([1, 336, 1, 1]))
start = time.time()
output = m(x400)
end = time.time()
print(end-start)
