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
        self.conv2d122 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()

    def forward(self, x384, x383):
        x385=self.conv2d121(x384)
        x386=self.relu95(x385)
        x387=self.conv2d122(x386)
        x388=self.sigmoid23(x387)
        x389=operator.mul(x388, x383)
        return x389

m = M().eval()
x384 = torch.randn(torch.Size([1, 2904, 1, 1]))
x383 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x384, x383)
end = time.time()
print(end-start)
