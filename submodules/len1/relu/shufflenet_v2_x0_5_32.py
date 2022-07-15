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
        self.relu32 = ReLU(inplace=True)

    def forward(self, x323):
        x324=self.relu32(x323)
        return x324

m = M().eval()
x323 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x323)
end = time.time()
print(end-start)
