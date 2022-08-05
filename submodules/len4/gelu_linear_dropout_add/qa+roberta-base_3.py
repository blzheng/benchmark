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
        self.linear23 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout12 = Dropout(p=0.1, inplace=False)

    def forward(self, x191, x190):
        x192=torch._C._nn.gelu(x191)
        x193=self.linear23(x192)
        x194=self.dropout12(x193)
        x195=operator.add(x194, x190)
        return x195

m = M().eval()
x191 = torch.randn(torch.Size([1, 384, 3072]))
x190 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x191, x190)
end = time.time()
print(end-start)
