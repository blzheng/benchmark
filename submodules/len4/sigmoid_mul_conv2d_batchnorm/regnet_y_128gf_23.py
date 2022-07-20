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
        self.sigmoid23 = Sigmoid()
        self.conv2d123 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d75 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x387, x383):
        x388=self.sigmoid23(x387)
        x389=operator.mul(x388, x383)
        x390=self.conv2d123(x389)
        x391=self.batchnorm2d75(x390)
        return x391

m = M().eval()
x387 = torch.randn(torch.Size([1, 2904, 1, 1]))
x383 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x387, x383)
end = time.time()
print(end-start)
