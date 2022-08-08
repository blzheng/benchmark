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
        self.relu43 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x157):
        x158=self.relu43(x157)
        x159=self.conv2d48(x158)
        x160=self.batchnorm2d48(x159)
        return x160

m = M().eval()
x157 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x157)
end = time.time()
print(end-start)
