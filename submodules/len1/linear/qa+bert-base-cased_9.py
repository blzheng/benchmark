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
        self.linear9 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x102):
        x103=self.linear9(x102)
        return x103

m = M().eval()
x102 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)
