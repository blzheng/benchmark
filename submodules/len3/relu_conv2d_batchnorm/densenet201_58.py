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
        self.relu119 = ReLU(inplace=True)
        self.conv2d119 = Conv2d(1536, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d120 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x422):
        x423=self.relu119(x422)
        x424=self.conv2d119(x423)
        x425=self.batchnorm2d120(x424)
        return x425

m = M().eval()
x422 = torch.randn(torch.Size([1, 1536, 14, 14]))
start = time.time()
output = m(x422)
end = time.time()
print(end-start)
