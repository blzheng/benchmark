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
        self.sigmoid54 = Sigmoid()

    def forward(self, x851, x847):
        x852=self.sigmoid54(x851)
        x853=operator.mul(x852, x847)
        return x853

m = M().eval()
x851 = torch.randn(torch.Size([1, 3840, 1, 1]))
x847 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x851, x847)
end = time.time()
print(end-start)
