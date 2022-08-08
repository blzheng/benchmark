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
        self.conv2d222 = Conv2d(3456, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d132 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d223 = Conv2d(576, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x660, x656, x650):
        x661=x660.sigmoid()
        x662=operator.mul(x656, x661)
        x663=self.conv2d222(x662)
        x664=self.batchnorm2d132(x663)
        x665=operator.add(x664, x650)
        x666=self.conv2d223(x665)
        return x666

m = M().eval()
x660 = torch.randn(torch.Size([1, 3456, 1, 1]))
x656 = torch.randn(torch.Size([1, 3456, 7, 7]))
x650 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x660, x656, x650)
end = time.time()
print(end-start)
