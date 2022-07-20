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
        self.conv2d57 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()

    def forward(self, x182):
        x183=self.conv2d57(x182)
        x184=self.sigmoid7(x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 40, 1, 1]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
