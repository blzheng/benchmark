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
        self.sigmoid24 = Sigmoid()
        self.conv2d121 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x377, x373):
        x378=self.sigmoid24(x377)
        x379=operator.mul(x378, x373)
        x380=self.conv2d121(x379)
        return x380

m = M().eval()
x377 = torch.randn(torch.Size([1, 960, 1, 1]))
x373 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x377, x373)
end = time.time()
print(end-start)
