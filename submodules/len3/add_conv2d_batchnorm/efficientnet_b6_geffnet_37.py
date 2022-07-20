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
        self.conv2d223 = Conv2d(576, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d133 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x664, x650):
        x665=operator.add(x664, x650)
        x666=self.conv2d223(x665)
        x667=self.batchnorm2d133(x666)
        return x667

m = M().eval()
x664 = torch.randn(torch.Size([1, 576, 7, 7]))
x650 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x664, x650)
end = time.time()
print(end-start)
