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
        self.batchnorm2d18 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(1056, 1056, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d19 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x88):
        x89=self.batchnorm2d18(x88)
        x90=self.relu21(x89)
        x91=self.conv2d29(x90)
        x92=self.batchnorm2d19(x91)
        return x92

m = M().eval()
x88 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
