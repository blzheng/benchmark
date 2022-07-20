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
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)
        self.relu14 = ReLU(inplace=True)

    def forward(self, x49):
        x50=self.linear1(x49)
        x51=self.relu14(x50)
        return x51

m = M().eval()
x49 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
