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
        self.conv2d87 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x278):
        x279=self.conv2d87(x278)
        x280=self.sigmoid13(x279)
        return x280

m = M().eval()
x278 = torch.randn(torch.Size([1, 40, 1, 1]))
start = time.time()
output = m(x278)
end = time.time()
print(end-start)
