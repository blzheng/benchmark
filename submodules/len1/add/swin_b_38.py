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

    def forward(self, x456, x470):
        x471=operator.add(x456, x470)
        return x471

m = M().eval()
x456 = torch.randn(torch.Size([1, 14, 14, 512]))
x470 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x456, x470)
end = time.time()
print(end-start)
