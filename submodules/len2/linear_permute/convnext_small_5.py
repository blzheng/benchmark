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
        self.linear11 = Linear(in_features=768, out_features=192, bias=True)

    def forward(self, x73):
        x74=self.linear11(x73)
        x75=torch.permute(x74, [0, 3, 1, 2])
        return x75

m = M().eval()
x73 = torch.randn(torch.Size([1, 28, 28, 768]))
start = time.time()
output = m(x73)
end = time.time()
print(end-start)
