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
        self.conv2d41 = Conv2d(264, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d42 = Conv2d(1056, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x128, x125):
        x129=self.conv2d41(x128)
        x130=self.sigmoid7(x129)
        x131=operator.mul(x130, x125)
        x132=self.conv2d42(x131)
        x133=self.batchnorm2d26(x132)
        return x133

m = M().eval()
x128 = torch.randn(torch.Size([1, 264, 1, 1]))
x125 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x128, x125)
end = time.time()
print(end-start)
