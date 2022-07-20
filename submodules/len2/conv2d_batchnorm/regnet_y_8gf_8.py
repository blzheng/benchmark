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
        self.conv2d12 = Conv2d(224, 448, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d8 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x37):
        x38=self.conv2d12(x37)
        x39=self.batchnorm2d8(x38)
        return x39

m = M().eval()
x37 = torch.randn(torch.Size([1, 224, 56, 56]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
