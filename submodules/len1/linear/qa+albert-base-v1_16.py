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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x131):
        x132=self.linear4(x131)
        return x132

m = M().eval()
x131 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)
