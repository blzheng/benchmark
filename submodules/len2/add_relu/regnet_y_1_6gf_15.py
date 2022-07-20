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

    def forward(self, x249, x263):
        x264=operator.add(x249, x263)
        x265=self.relu64(x264)
        return x265

m = M().eval()
x249 = torch.randn(torch.Size([1, 336, 14, 14]))
x263 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x249, x263)
end = time.time()
print(end-start)
