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
        self.conv2d142 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x418, x423):
        x424=operator.mul(x418, x423)
        x425=self.conv2d142(x424)
        x426=self.batchnorm2d84(x425)
        return x426

m = M().eval()
x418 = torch.randn(torch.Size([1, 1200, 14, 14]))
x423 = torch.randn(torch.Size([1, 1200, 1, 1]))
start = time.time()
output = m(x418, x423)
end = time.time()
print(end-start)
