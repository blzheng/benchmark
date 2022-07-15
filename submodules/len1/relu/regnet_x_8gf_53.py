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
        self.relu53 = ReLU(inplace=True)

    def forward(self, x184):
        x185=self.relu53(x184)
        return x185

m = M().eval()
x184 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
