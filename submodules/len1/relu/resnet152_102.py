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
        self.relu100 = ReLU(inplace=True)

    def forward(self, x349):
        x350=self.relu100(x349)
        return x350

m = M().eval()
x349 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x349)
end = time.time()
print(end-start)
