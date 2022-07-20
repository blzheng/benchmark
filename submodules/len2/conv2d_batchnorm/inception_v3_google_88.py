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
        self.conv2d88 = Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
        self.batchnorm2d88 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x298):
        x302=self.conv2d88(x298)
        x303=self.batchnorm2d88(x302)
        return x303

m = M().eval()
x298 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x298)
end = time.time()
print(end-start)
