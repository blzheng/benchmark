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
        self.conv2d241 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d143 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x759):
        x760=self.conv2d241(x759)
        x761=self.batchnorm2d143(x760)
        return x761

m = M().eval()
x759 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x759)
end = time.time()
print(end-start)
