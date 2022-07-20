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
        self.batchnorm2d74 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu74 = ReLU(inplace=True)
        self.conv2d74 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x263):
        x264=self.batchnorm2d74(x263)
        x265=self.relu74(x264)
        x266=self.conv2d74(x265)
        return x266

m = M().eval()
x263 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x263)
end = time.time()
print(end-start)
