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
        self.relu117 = ReLU(inplace=True)

    def forward(self, x415):
        x416=self.relu117(x415)
        return x416

m = M().eval()
x415 = torch.randn(torch.Size([1, 1504, 14, 14]))
start = time.time()
output = m(x415)
end = time.time()
print(end-start)
