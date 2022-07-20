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
        self.conv2d127 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()

    def forward(self, x402):
        x403=self.conv2d127(x402)
        x404=self.sigmoid24(x403)
        return x404

m = M().eval()
x402 = torch.randn(torch.Size([1, 84, 1, 1]))
start = time.time()
output = m(x402)
end = time.time()
print(end-start)
