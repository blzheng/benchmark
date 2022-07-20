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
        self.conv2d132 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x395, x381):
        x396=operator.add(x395, x381)
        x397=self.conv2d132(x396)
        x398=self.batchnorm2d78(x397)
        return x398

m = M().eval()
x395 = torch.randn(torch.Size([1, 160, 14, 14]))
x381 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x395, x381)
end = time.time()
print(end-start)
