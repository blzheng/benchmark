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
        self.linear69 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x491):
        x499=self.linear69(x491)
        return x499

m = M().eval()
x491 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x491)
end = time.time()
print(end-start)
