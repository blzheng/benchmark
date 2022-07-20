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
        self.sigmoid3 = Sigmoid()

    def forward(self, x51, x47):
        x52=self.sigmoid3(x51)
        x53=operator.mul(x52, x47)
        return x53

m = M().eval()
x51 = torch.randn(torch.Size([1, 144, 1, 1]))
x47 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x51, x47)
end = time.time()
print(end-start)
