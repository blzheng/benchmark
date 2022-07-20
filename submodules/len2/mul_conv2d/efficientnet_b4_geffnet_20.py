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
        self.conv2d103 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x300, x305):
        x306=operator.mul(x300, x305)
        x307=self.conv2d103(x306)
        return x307

m = M().eval()
x300 = torch.randn(torch.Size([1, 960, 14, 14]))
x305 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x300, x305)
end = time.time()
print(end-start)
