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
        self.relu76 = ReLU(inplace=True)
        self.conv2d82 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x269):
        x270=self.relu76(x269)
        x271=self.conv2d82(x270)
        return x271

m = M().eval()
x269 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x269)
end = time.time()
print(end-start)
