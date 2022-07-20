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
        self.relu63 = ReLU()
        self.conv2d83 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()

    def forward(self, x259):
        x260=self.relu63(x259)
        x261=self.conv2d83(x260)
        x262=self.sigmoid15(x261)
        return x262

m = M().eval()
x259 = torch.randn(torch.Size([1, 110, 1, 1]))
start = time.time()
output = m(x259)
end = time.time()
print(end-start)
