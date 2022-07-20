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
        self.conv2d112 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x356, x353):
        x357=self.conv2d112(x356)
        x358=self.sigmoid18(x357)
        x359=operator.mul(x358, x353)
        return x359

m = M().eval()
x356 = torch.randn(torch.Size([1, 64, 1, 1]))
x353 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x356, x353)
end = time.time()
print(end-start)