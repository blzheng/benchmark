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
        self.conv2d43 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x157):
        x158=self.conv2d43(x157)
        x159=self.batchnorm2d43(x158)
        return x159

m = M().eval()
x157 = torch.randn(torch.Size([1, 32, 14, 14]))
start = time.time()
output = m(x157)
end = time.time()
print(end-start)
