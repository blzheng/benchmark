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
        self.conv2d103 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x303, x299):
        x304=x303.sigmoid()
        x305=operator.mul(x299, x304)
        x306=self.conv2d103(x305)
        return x306

m = M().eval()
x303 = torch.randn(torch.Size([1, 1392, 1, 1]))
x299 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x303, x299)
end = time.time()
print(end-start)
