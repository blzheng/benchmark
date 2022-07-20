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
        self.conv2d43 = Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x131):
        x132=self.conv2d43(x131)
        x133=self.batchnorm2d25(x132)
        return x133

m = M().eval()
x131 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)
