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
        self.sigmoid10 = Sigmoid()

    def forward(self, x179, x175):
        x180=self.sigmoid10(x179)
        x181=operator.mul(x180, x175)
        return x181

m = M().eval()
x179 = torch.randn(torch.Size([1, 2904, 1, 1]))
x175 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x179, x175)
end = time.time()
print(end-start)
