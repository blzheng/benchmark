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
        self.sigmoid13 = Sigmoid()
        self.conv2d67 = Conv2d(432, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x206, x202):
        x207=self.sigmoid13(x206)
        x208=operator.mul(x207, x202)
        x209=self.conv2d67(x208)
        return x209

m = M().eval()
x206 = torch.randn(torch.Size([1, 432, 1, 1]))
x202 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x206, x202)
end = time.time()
print(end-start)
