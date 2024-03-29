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
        self.conv2d47 = Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x171):
        x172=self.conv2d47(x171)
        x173=self.batchnorm2d48(x172)
        return x173

m = M().eval()
x171 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x171)
end = time.time()
print(end-start)
