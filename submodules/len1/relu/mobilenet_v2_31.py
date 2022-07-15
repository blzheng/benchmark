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
        self.relu631 = ReLU6(inplace=True)

    def forward(self, x134):
        x135=self.relu631(x134)
        return x135

m = M().eval()
x134 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x134)
end = time.time()
print(end-start)
