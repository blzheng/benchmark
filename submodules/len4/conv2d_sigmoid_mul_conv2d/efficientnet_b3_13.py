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
        self.conv2d67 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d68 = Conv2d(576, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x206, x203):
        x207=self.conv2d67(x206)
        x208=self.sigmoid13(x207)
        x209=operator.mul(x208, x203)
        x210=self.conv2d68(x209)
        return x210

m = M().eval()
x206 = torch.randn(torch.Size([1, 24, 1, 1]))
x203 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x206, x203)
end = time.time()
print(end-start)
