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
        self.batchnorm2d9 = BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
        self.batchnorm2d10 = BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)

    def forward(self, x30):
        x31=self.batchnorm2d9(x30)
        x32=self.relu6(x31)
        x33=self.conv2d10(x32)
        x34=self.batchnorm2d10(x33)
        x35=self.relu7(x34)
        return x35

m = M().eval()
x30 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
