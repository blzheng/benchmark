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
        self.sigmoid25 = Sigmoid()

    def forward(self, x469, x465):
        x470=self.sigmoid25(x469)
        x471=operator.mul(x470, x465)
        return x471

m = M().eval()
x469 = torch.randn(torch.Size([1, 1536, 1, 1]))
x465 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x469, x465)
end = time.time()
print(end-start)
