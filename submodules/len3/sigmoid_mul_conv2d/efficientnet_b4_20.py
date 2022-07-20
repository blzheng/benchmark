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
        self.sigmoid20 = Sigmoid()
        self.conv2d103 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x317, x313):
        x318=self.sigmoid20(x317)
        x319=operator.mul(x318, x313)
        x320=self.conv2d103(x319)
        return x320

m = M().eval()
x317 = torch.randn(torch.Size([1, 960, 1, 1]))
x313 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x317, x313)
end = time.time()
print(end-start)
