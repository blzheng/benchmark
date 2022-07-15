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
        self.relu70 = ReLU(inplace=True)

    def forward(self, x288):
        x289=self.relu70(x288)
        return x289

m = M().eval()
x288 = torch.randn(torch.Size([1, 3024, 7, 7]))
start = time.time()
output = m(x288)
end = time.time()
print(end-start)
