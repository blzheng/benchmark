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
        self.sigmoid7 = Sigmoid()

    def forward(self, x129, x125):
        x130=self.sigmoid7(x129)
        x131=operator.mul(x130, x125)
        return x131

m = M().eval()
x129 = torch.randn(torch.Size([1, 120, 1, 1]))
x125 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x129, x125)
end = time.time()
print(end-start)
