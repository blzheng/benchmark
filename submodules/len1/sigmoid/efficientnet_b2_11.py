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
        self.sigmoid11 = Sigmoid()

    def forward(self, x175):
        x176=self.sigmoid11(x175)
        return x176

m = M().eval()
x175 = torch.randn(torch.Size([1, 528, 1, 1]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
