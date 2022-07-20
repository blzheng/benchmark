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
        self.sigmoid39 = Sigmoid()

    def forward(self, x745):
        x746=self.sigmoid39(x745)
        return x746

m = M().eval()
x745 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x745)
end = time.time()
print(end-start)