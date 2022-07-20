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
        self.conv2d117 = Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()

    def forward(self, x362, x359):
        x363=self.conv2d117(x362)
        x364=self.sigmoid23(x363)
        x365=operator.mul(x364, x359)
        return x365

m = M().eval()
x362 = torch.randn(torch.Size([1, 68, 1, 1]))
x359 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x362, x359)
end = time.time()
print(end-start)
