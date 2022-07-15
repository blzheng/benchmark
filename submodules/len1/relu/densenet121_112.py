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
        self.relu112 = ReLU(inplace=True)

    def forward(self, x399):
        x400=self.relu112(x399)
        return x400

m = M().eval()
x399 = torch.randn(torch.Size([1, 896, 7, 7]))
start = time.time()
output = m(x399)
end = time.time()
print(end-start)
