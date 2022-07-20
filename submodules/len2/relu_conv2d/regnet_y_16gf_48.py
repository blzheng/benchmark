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
        self.relu64 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x264):
        x265=self.relu64(x264)
        x266=self.conv2d84(x265)
        return x266

m = M().eval()
x264 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x264)
end = time.time()
print(end-start)
