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
        self.linear59 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x343):
        x344=self.linear59(x343)
        x345=torch.permute(x344, [0, 3, 1, 2])
        return x345

m = M().eval()
x343 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x343)
end = time.time()
print(end-start)
