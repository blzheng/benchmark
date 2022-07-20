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
        self.sigmoid19 = Sigmoid()
        self.conv2d98 = Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x299, x295):
        x300=self.sigmoid19(x299)
        x301=operator.mul(x300, x295)
        x302=self.conv2d98(x301)
        return x302

m = M().eval()
x299 = torch.randn(torch.Size([1, 1248, 1, 1]))
x295 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x299, x295)
end = time.time()
print(end-start)
