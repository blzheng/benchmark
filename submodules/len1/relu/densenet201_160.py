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
        self.relu160 = ReLU(inplace=True)

    def forward(self, x567):
        x568=self.relu160(x567)
        return x568

m = M().eval()
x567 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x567)
end = time.time()
print(end-start)
