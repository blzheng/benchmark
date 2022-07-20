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
        self.batchnorm2d39 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x172):
        x173=self.batchnorm2d39(x172)
        x174=self.conv2d54(x173)
        return x174

m = M().eval()
x172 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x172)
end = time.time()
print(end-start)
