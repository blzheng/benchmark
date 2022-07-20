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
        self.conv2d270 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid54 = Sigmoid()

    def forward(self, x850):
        x851=self.conv2d270(x850)
        x852=self.sigmoid54(x851)
        return x852

m = M().eval()
x850 = torch.randn(torch.Size([1, 160, 1, 1]))
start = time.time()
output = m(x850)
end = time.time()
print(end-start)
