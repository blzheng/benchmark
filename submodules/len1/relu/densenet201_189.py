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
        self.relu189 = ReLU(inplace=True)

    def forward(self, x668):
        x669=self.relu189(x668)
        return x669

m = M().eval()
x668 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x668)
end = time.time()
print(end-start)
