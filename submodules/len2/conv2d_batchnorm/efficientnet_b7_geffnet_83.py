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
        self.conv2d141 = Conv2d(960, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d83 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x423):
        x424=self.conv2d141(x423)
        x425=self.batchnorm2d83(x424)
        return x425

m = M().eval()
x423 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x423)
end = time.time()
print(end-start)
