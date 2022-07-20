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
        self.conv2d66 = Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x205, x202):
        x206=self.conv2d66(x205)
        x207=self.sigmoid13(x206)
        x208=operator.mul(x207, x202)
        return x208

m = M().eval()
x205 = torch.randn(torch.Size([1, 16, 1, 1]))
x202 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x205, x202)
end = time.time()
print(end-start)
