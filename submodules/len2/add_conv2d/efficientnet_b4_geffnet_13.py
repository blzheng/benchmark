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
        self.conv2d94 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x278, x264):
        x279=operator.add(x278, x264)
        x280=self.conv2d94(x279)
        return x280

m = M().eval()
x278 = torch.randn(torch.Size([1, 160, 14, 14]))
x264 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x278, x264)
end = time.time()
print(end-start)
