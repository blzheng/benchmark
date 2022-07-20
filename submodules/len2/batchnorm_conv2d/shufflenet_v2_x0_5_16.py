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
        self.batchnorm2d47 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x303):
        x304=self.batchnorm2d47(x303)
        x305=self.conv2d48(x304)
        return x305

m = M().eval()
x303 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x303)
end = time.time()
print(end-start)
