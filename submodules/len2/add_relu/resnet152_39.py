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
        self.relu118 = ReLU(inplace=True)

    def forward(self, x408, x400):
        x409=operator.add(x408, x400)
        x410=self.relu118(x409)
        return x410

m = M().eval()
x408 = torch.randn(torch.Size([1, 1024, 14, 14]))
x400 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x408, x400)
end = time.time()
print(end-start)
