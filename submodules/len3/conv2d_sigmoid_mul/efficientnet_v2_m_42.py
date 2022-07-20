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
        self.conv2d237 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid42 = Sigmoid()

    def forward(self, x757, x754):
        x758=self.conv2d237(x757)
        x759=self.sigmoid42(x758)
        x760=operator.mul(x759, x754)
        return x760

m = M().eval()
x757 = torch.randn(torch.Size([1, 128, 1, 1]))
x754 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x757, x754)
end = time.time()
print(end-start)
