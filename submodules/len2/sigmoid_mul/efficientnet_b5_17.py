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
        self.sigmoid17 = Sigmoid()

    def forward(self, x268, x264):
        x269=self.sigmoid17(x268)
        x270=operator.mul(x269, x264)
        return x270

m = M().eval()
x268 = torch.randn(torch.Size([1, 768, 1, 1]))
x264 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x268, x264)
end = time.time()
print(end-start)
