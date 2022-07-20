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
        self.batchnorm2d52 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x168, x161):
        x169=self.batchnorm2d52(x168)
        x170=operator.add(x161, x169)
        x171=self.relu48(x170)
        x172=self.conv2d53(x171)
        x173=self.batchnorm2d53(x172)
        return x173

m = M().eval()
x168 = torch.randn(torch.Size([1, 400, 7, 7]))
x161 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x168, x161)
end = time.time()
print(end-start)
