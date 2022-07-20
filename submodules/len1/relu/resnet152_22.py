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
        self.relu22 = ReLU(inplace=True)

    def forward(self, x80):
        x81=self.relu22(x80)
        return x81

m = M().eval()
x80 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)