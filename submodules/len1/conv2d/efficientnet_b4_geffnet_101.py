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
        self.conv2d101 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x301):
        x302=self.conv2d101(x301)
        return x302

m = M().eval()
x301 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x301)
end = time.time()
print(end-start)
