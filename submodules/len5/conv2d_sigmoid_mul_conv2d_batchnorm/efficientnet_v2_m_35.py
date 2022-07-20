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
        self.conv2d202 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid35 = Sigmoid()
        self.conv2d203 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d131 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x647, x644):
        x648=self.conv2d202(x647)
        x649=self.sigmoid35(x648)
        x650=operator.mul(x649, x644)
        x651=self.conv2d203(x650)
        x652=self.batchnorm2d131(x651)
        return x652

m = M().eval()
x647 = torch.randn(torch.Size([1, 76, 1, 1]))
x644 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x647, x644)
end = time.time()
print(end-start)
