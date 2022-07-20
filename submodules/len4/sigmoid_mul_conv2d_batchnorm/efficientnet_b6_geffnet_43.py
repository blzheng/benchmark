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
        self.conv2d217 = Conv2d(3456, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d129 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x645, x641):
        x646=x645.sigmoid()
        x647=operator.mul(x641, x646)
        x648=self.conv2d217(x647)
        x649=self.batchnorm2d129(x648)
        return x649

m = M().eval()
x645 = torch.randn(torch.Size([1, 3456, 1, 1]))
x641 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x645, x641)
end = time.time()
print(end-start)
