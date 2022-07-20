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
        self.conv2d129 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d129 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x426, x420):
        x427=self.conv2d129(x426)
        x428=self.batchnorm2d129(x427)
        x429=operator.add(x428, x420)
        return x429

m = M().eval()
x426 = torch.randn(torch.Size([1, 256, 14, 14]))
x420 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x426, x420)
end = time.time()
print(end-start)
