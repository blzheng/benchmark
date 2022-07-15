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
        self.linear0 = Linear(in_features=3024, out_features=1000, bias=True)

    def forward(self, x301):
        x302=self.linear0(x301)
        return x302

m = M().eval()
x301 = torch.randn(torch.Size([1, 3024]))
start = time.time()
output = m(x301)
end = time.time()
print(end-start)
