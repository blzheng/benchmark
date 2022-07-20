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
        self.batchnorm2d74 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu74 = ReLU(inplace=True)

    def forward(self, x263):
        x264=self.batchnorm2d74(x263)
        x265=self.relu74(x264)
        return x265

m = M().eval()
x263 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x263)
end = time.time()
print(end-start)
