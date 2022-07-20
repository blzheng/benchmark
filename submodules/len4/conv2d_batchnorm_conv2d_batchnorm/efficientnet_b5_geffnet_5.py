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
        self.conv2d182 = Conv2d(1824, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d108 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d183 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d109 = BatchNorm2d(3072, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x543):
        x544=self.conv2d182(x543)
        x545=self.batchnorm2d108(x544)
        x546=self.conv2d183(x545)
        x547=self.batchnorm2d109(x546)
        return x547

m = M().eval()
x543 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x543)
end = time.time()
print(end-start)
