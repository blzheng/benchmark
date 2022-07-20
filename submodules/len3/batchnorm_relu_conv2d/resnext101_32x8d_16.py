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
        self.batchnorm2d28 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)

    def forward(self, x91):
        x92=self.batchnorm2d28(x91)
        x93=self.relu25(x92)
        x94=self.conv2d29(x93)
        return x94

m = M().eval()
x91 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
