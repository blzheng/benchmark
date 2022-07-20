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
        self.conv2d87 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x258, x254):
        x259=x258.sigmoid()
        x260=operator.mul(x254, x259)
        x261=self.conv2d87(x260)
        x262=self.batchnorm2d51(x261)
        return x262

m = M().eval()
x258 = torch.randn(torch.Size([1, 864, 1, 1]))
x254 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x258, x254)
end = time.time()
print(end-start)
