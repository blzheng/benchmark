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
        self.conv2d48 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)

    def forward(self, x156):
        x157=self.conv2d48(x156)
        x158=self.batchnorm2d48(x157)
        x159=self.relu32(x158)
        return x159

m = M().eval()
x156 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
