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
        self.relu39 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)
        self.batchnorm2d59 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x190):
        x191=self.relu39(x190)
        x192=self.conv2d59(x191)
        x193=self.batchnorm2d59(x192)
        return x193

m = M().eval()
x190 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x190)
end = time.time()
print(end-start)
