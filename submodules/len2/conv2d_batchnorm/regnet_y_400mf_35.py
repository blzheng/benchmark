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
        self.conv2d55 = Conv2d(208, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x169):
        x172=self.conv2d55(x169)
        x173=self.batchnorm2d35(x172)
        return x173

m = M().eval()
x169 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
