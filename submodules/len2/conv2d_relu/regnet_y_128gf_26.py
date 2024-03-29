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
        self.conv2d137 = Conv2d(7392, 726, kernel_size=(1, 1), stride=(1, 1))
        self.relu107 = ReLU()

    def forward(self, x434):
        x435=self.conv2d137(x434)
        x436=self.relu107(x435)
        return x436

m = M().eval()
x434 = torch.randn(torch.Size([1, 7392, 1, 1]))
start = time.time()
output = m(x434)
end = time.time()
print(end-start)
