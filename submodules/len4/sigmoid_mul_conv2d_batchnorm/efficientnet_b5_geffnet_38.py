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
        self.conv2d192 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d114 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x570, x566):
        x571=x570.sigmoid()
        x572=operator.mul(x566, x571)
        x573=self.conv2d192(x572)
        x574=self.batchnorm2d114(x573)
        return x574

m = M().eval()
x570 = torch.randn(torch.Size([1, 3072, 1, 1]))
x566 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x570, x566)
end = time.time()
print(end-start)
