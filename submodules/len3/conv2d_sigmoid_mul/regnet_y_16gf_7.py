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
        self.conv2d42 = Conv2d(308, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()

    def forward(self, x130, x127):
        x131=self.conv2d42(x130)
        x132=self.sigmoid7(x131)
        x133=operator.mul(x132, x127)
        return x133

m = M().eval()
x130 = torch.randn(torch.Size([1, 308, 1, 1]))
x127 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x130, x127)
end = time.time()
print(end-start)
