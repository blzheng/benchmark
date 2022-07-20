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
        self.conv2d122 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d123 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d83 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x393, x390):
        x394=self.conv2d122(x393)
        x395=self.sigmoid19(x394)
        x396=operator.mul(x395, x390)
        x397=self.conv2d123(x396)
        x398=self.batchnorm2d83(x397)
        return x398

m = M().eval()
x393 = torch.randn(torch.Size([1, 44, 1, 1]))
x390 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x393, x390)
end = time.time()
print(end-start)
