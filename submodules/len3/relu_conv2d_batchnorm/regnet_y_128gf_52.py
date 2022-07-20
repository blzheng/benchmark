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
        self.relu104 = ReLU(inplace=True)
        self.conv2d134 = Conv2d(2904, 7392, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d82 = BatchNorm2d(7392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x424):
        x425=self.relu104(x424)
        x426=self.conv2d134(x425)
        x427=self.batchnorm2d82(x426)
        return x427

m = M().eval()
x424 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x424)
end = time.time()
print(end-start)
