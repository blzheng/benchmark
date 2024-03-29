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
        self.gelu3 = GELU(approximate='none')

    def forward(self, x97):
        x98=self.gelu3(x97)
        return x98

m = M().eval()
x97 = torch.randn(torch.Size([1, 28, 28, 1024]))
start = time.time()
output = m(x97)
end = time.time()
print(end-start)
