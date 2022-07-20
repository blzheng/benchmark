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
        self.conv2d59 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)
        self.batchnorm2d59 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)

    def forward(self, x191):
        x192=self.conv2d59(x191)
        x193=self.batchnorm2d59(x192)
        x194=self.relu40(x193)
        return x194

m = M().eval()
x191 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x191)
end = time.time()
print(end-start)
