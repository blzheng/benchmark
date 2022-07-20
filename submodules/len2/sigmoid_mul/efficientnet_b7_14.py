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
        self.sigmoid14 = Sigmoid()

    def forward(self, x219, x215):
        x220=self.sigmoid14(x219)
        x221=operator.mul(x220, x215)
        return x221

m = M().eval()
x219 = torch.randn(torch.Size([1, 480, 1, 1]))
x215 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x219, x215)
end = time.time()
print(end-start)
