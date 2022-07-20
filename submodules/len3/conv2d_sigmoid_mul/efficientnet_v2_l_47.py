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
        self.conv2d271 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid47 = Sigmoid()

    def forward(self, x872, x869):
        x873=self.conv2d271(x872)
        x874=self.sigmoid47(x873)
        x875=operator.mul(x874, x869)
        return x875

m = M().eval()
x872 = torch.randn(torch.Size([1, 96, 1, 1]))
x869 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x872, x869)
end = time.time()
print(end-start)
