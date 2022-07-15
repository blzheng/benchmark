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
        self.relu105 = ReLU(inplace=True)

    def forward(self, x429):
        x430=self.relu105(x429)
        return x430

m = M().eval()
x429 = torch.randn(torch.Size([1, 7392, 14, 14]))
start = time.time()
output = m(x429)
end = time.time()
print(end-start)
