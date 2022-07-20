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
        self.conv2d156 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()
        self.conv2d157 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d107 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x506, x503):
        x507=self.conv2d156(x506)
        x508=self.sigmoid24(x507)
        x509=operator.mul(x508, x503)
        x510=self.conv2d157(x509)
        x511=self.batchnorm2d107(x510)
        return x511

m = M().eval()
x506 = torch.randn(torch.Size([1, 56, 1, 1]))
x503 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x506, x503)
end = time.time()
print(end-start)
