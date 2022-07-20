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
        self.conv2d56 = Conv2d(440, 440, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=55, bias=False)
        self.batchnorm2d36 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)

    def forward(self, x174):
        x175=self.conv2d56(x174)
        x176=self.batchnorm2d36(x175)
        x177=self.relu42(x176)
        return x177

m = M().eval()
x174 = torch.randn(torch.Size([1, 440, 14, 14]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
