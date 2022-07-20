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
        self.conv2d27 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()

    def forward(self, x82):
        x83=self.conv2d27(x82)
        x84=self.sigmoid5(x83)
        return x84

m = M().eval()
x82 = torch.randn(torch.Size([1, 8, 1, 1]))
start = time.time()
output = m(x82)
end = time.time()
print(end-start)
