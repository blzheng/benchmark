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
        self.conv2d182 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d108 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x544, x530):
        x545=operator.add(x544, x530)
        x546=self.conv2d182(x545)
        x547=self.batchnorm2d108(x546)
        return x547

m = M().eval()
x544 = torch.randn(torch.Size([1, 224, 14, 14]))
x530 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x544, x530)
end = time.time()
print(end-start)