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
        self.conv2d51 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()

    def forward(self, x172, x169):
        x173=self.conv2d51(x172)
        x174=self.sigmoid3(x173)
        x175=operator.mul(x174, x169)
        return x175

m = M().eval()
x172 = torch.randn(torch.Size([1, 48, 1, 1]))
x169 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x172, x169)
end = time.time()
print(end-start)
