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
        self.conv2d30 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(480, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x85, x77):
        x86=operator.add(x85, x77)
        x87=self.conv2d30(x86)
        x88=self.batchnorm2d30(x87)
        return x88

m = M().eval()
x85 = torch.randn(torch.Size([1, 80, 14, 14]))
x77 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x85, x77)
end = time.time()
print(end-start)
