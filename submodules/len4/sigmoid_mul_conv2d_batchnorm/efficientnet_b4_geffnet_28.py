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
        self.conv2d143 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x423, x419):
        x424=x423.sigmoid()
        x425=operator.mul(x419, x424)
        x426=self.conv2d143(x425)
        x427=self.batchnorm2d85(x426)
        return x427

m = M().eval()
x423 = torch.randn(torch.Size([1, 1632, 1, 1]))
x419 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x423, x419)
end = time.time()
print(end-start)
