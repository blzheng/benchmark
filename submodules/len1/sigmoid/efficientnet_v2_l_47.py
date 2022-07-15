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
        self.sigmoid47 = Sigmoid()

    def forward(self, x873):
        x874=self.sigmoid47(x873)
        return x874

m = M().eval()
x873 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x873)
end = time.time()
print(end-start)
