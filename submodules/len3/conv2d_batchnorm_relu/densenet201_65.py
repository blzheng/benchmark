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
        self.conv2d131 = Conv2d(1728, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d132 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu132 = ReLU(inplace=True)

    def forward(self, x465):
        x466=self.conv2d131(x465)
        x467=self.batchnorm2d132(x466)
        x468=self.relu132(x467)
        return x468

m = M().eval()
x465 = torch.randn(torch.Size([1, 1728, 14, 14]))
start = time.time()
output = m(x465)
end = time.time()
print(end-start)
