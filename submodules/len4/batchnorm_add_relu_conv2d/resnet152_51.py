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
        self.batchnorm2d148 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu142 = ReLU(inplace=True)
        self.conv2d149 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x489, x488):
        x490=self.batchnorm2d148(x489)
        x491=operator.add(x488, x490)
        x492=self.relu142(x491)
        x493=self.conv2d149(x492)
        return x493

m = M().eval()
x489 = torch.randn(torch.Size([1, 2048, 7, 7]))
x488 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x489, x488)
end = time.time()
print(end-start)
