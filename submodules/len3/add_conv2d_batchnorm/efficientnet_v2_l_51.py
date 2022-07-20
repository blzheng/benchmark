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
        self.conv2d228 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d150 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x734, x719):
        x735=operator.add(x734, x719)
        x736=self.conv2d228(x735)
        x737=self.batchnorm2d150(x736)
        return x737

m = M().eval()
x734 = torch.randn(torch.Size([1, 384, 7, 7]))
x719 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x734, x719)
end = time.time()
print(end-start)
