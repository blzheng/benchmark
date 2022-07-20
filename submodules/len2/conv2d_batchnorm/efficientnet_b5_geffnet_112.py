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
        self.conv2d188 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d112 = BatchNorm2d(3072, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x560):
        x561=self.conv2d188(x560)
        x562=self.batchnorm2d112(x561)
        return x562

m = M().eval()
x560 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x560)
end = time.time()
print(end-start)
