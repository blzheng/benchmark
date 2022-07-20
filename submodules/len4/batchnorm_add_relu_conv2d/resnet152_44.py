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
        self.batchnorm2d129 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu124 = ReLU(inplace=True)
        self.conv2d130 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x427, x420):
        x428=self.batchnorm2d129(x427)
        x429=operator.add(x428, x420)
        x430=self.relu124(x429)
        x431=self.conv2d130(x430)
        return x431

m = M().eval()
x427 = torch.randn(torch.Size([1, 1024, 14, 14]))
x420 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x427, x420)
end = time.time()
print(end-start)
