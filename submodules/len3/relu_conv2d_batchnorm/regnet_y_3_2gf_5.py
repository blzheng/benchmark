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
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(216, 216, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d10 = BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x41):
        x42=self.relu9(x41)
        x43=self.conv2d14(x42)
        x44=self.batchnorm2d10(x43)
        return x44

m = M().eval()
x41 = torch.randn(torch.Size([1, 216, 56, 56]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)
