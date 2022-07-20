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
        self.conv2d106 = Conv2d(2904, 726, kernel_size=(1, 1), stride=(1, 1))
        self.relu83 = ReLU()
        self.conv2d107 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x336):
        x337=self.conv2d106(x336)
        x338=self.relu83(x337)
        x339=self.conv2d107(x338)
        return x339

m = M().eval()
x336 = torch.randn(torch.Size([1, 2904, 1, 1]))
start = time.time()
output = m(x336)
end = time.time()
print(end-start)
