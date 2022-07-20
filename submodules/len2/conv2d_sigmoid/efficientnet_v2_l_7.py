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
        self.conv2d71 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()

    def forward(self, x236):
        x237=self.conv2d71(x236)
        x238=self.sigmoid7(x237)
        return x238

m = M().eval()
x236 = torch.randn(torch.Size([1, 48, 1, 1]))
start = time.time()
output = m(x236)
end = time.time()
print(end-start)
