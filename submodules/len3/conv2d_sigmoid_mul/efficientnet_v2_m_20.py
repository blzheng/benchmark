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
        self.conv2d127 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()

    def forward(self, x409, x406):
        x410=self.conv2d127(x409)
        x411=self.sigmoid20(x410)
        x412=operator.mul(x411, x406)
        return x412

m = M().eval()
x409 = torch.randn(torch.Size([1, 44, 1, 1]))
x406 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x409, x406)
end = time.time()
print(end-start)
