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
        self.conv2d80 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()

    def forward(self, x250):
        x251=self.conv2d80(x250)
        x252=self.sigmoid16(x251)
        return x252

m = M().eval()
x250 = torch.randn(torch.Size([1, 20, 1, 1]))
start = time.time()
output = m(x250)
end = time.time()
print(end-start)
