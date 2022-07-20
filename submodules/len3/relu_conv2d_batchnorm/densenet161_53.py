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
        self.relu109 = ReLU(inplace=True)
        self.conv2d109 = Conv2d(2064, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x387):
        x388=self.relu109(x387)
        x389=self.conv2d109(x388)
        x390=self.batchnorm2d110(x389)
        return x390

m = M().eval()
x387 = torch.randn(torch.Size([1, 2064, 14, 14]))
start = time.time()
output = m(x387)
end = time.time()
print(end-start)
