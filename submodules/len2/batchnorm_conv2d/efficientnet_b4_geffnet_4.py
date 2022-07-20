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
        self.batchnorm2d67 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d114 = Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x337):
        x338=self.batchnorm2d67(x337)
        x339=self.conv2d114(x338)
        return x339

m = M().eval()
x337 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x337)
end = time.time()
print(end-start)
