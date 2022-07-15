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
        self.conv2d121 = Conv2d(2904, 726, kernel_size=(1, 1), stride=(1, 1))
        self.relu95 = ReLU()

    def forward(self, x384):
        x385=self.conv2d121(x384)
        x386=self.relu95(x385)
        return x386

m = M().eval()
x384 = torch.randn(torch.Size([1, 2904, 1, 1]))
start = time.time()
output = m(x384)
end = time.time()
print(end-start)
