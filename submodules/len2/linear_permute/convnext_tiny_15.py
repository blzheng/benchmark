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
        self.linear31 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x195):
        x196=self.linear31(x195)
        x197=torch.permute(x196, [0, 3, 1, 2])
        return x197

m = M().eval()
x195 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x195)
end = time.time()
print(end-start)
