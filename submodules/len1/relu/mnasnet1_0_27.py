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
        self.relu27 = ReLU(inplace=True)

    def forward(self, x116):
        x117=self.relu27(x116)
        return x117

m = M().eval()
x116 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x116)
end = time.time()
print(end-start)
