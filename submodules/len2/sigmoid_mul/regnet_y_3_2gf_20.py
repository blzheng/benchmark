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
        self.sigmoid20 = Sigmoid()

    def forward(self, x341, x337):
        x342=self.sigmoid20(x341)
        x343=operator.mul(x342, x337)
        return x343

m = M().eval()
x341 = torch.randn(torch.Size([1, 1512, 1, 1]))
x337 = torch.randn(torch.Size([1, 1512, 7, 7]))
start = time.time()
output = m(x341, x337)
end = time.time()
print(end-start)
