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
        self.conv2d67 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x206, x203):
        x207=self.conv2d67(x206)
        x208=self.sigmoid13(x207)
        x209=operator.mul(x208, x203)
        return x209

m = M().eval()
x206 = torch.randn(torch.Size([1, 28, 1, 1]))
x203 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x206, x203)
end = time.time()
print(end-start)
