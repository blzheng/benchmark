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
        self.batchnorm2d52 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)
        self.batchnorm2d53 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x169):
        x170=self.batchnorm2d52(x169)
        x171=self.relu35(x170)
        x172=self.conv2d53(x171)
        x173=self.batchnorm2d53(x172)
        return x173

m = M().eval()
x169 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
