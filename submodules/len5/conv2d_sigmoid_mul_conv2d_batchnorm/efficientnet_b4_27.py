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
        self.conv2d137 = Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()
        self.conv2d138 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x426, x423):
        x427=self.conv2d137(x426)
        x428=self.sigmoid27(x427)
        x429=operator.mul(x428, x423)
        x430=self.conv2d138(x429)
        x431=self.batchnorm2d82(x430)
        return x431

m = M().eval()
x426 = torch.randn(torch.Size([1, 68, 1, 1]))
x423 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x426, x423)
end = time.time()
print(end-start)
