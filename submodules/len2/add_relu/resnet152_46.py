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
        self.relu139 = ReLU(inplace=True)

    def forward(self, x478, x470):
        x479=operator.add(x478, x470)
        x480=self.relu139(x479)
        return x480

m = M().eval()
x478 = torch.randn(torch.Size([1, 1024, 14, 14]))
x470 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x478, x470)
end = time.time()
print(end-start)
