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
        self.conv2d228 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d146 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x727, x722):
        x728=operator.mul(x727, x722)
        x729=self.conv2d228(x728)
        x730=self.batchnorm2d146(x729)
        return x730

m = M().eval()
x727 = torch.randn(torch.Size([1, 3072, 1, 1]))
x722 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x727, x722)
end = time.time()
print(end-start)
