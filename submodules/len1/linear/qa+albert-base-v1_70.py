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

    def forward(self, x464):
        x465=self.linear4(x464)
        return x465

m = M().eval()
x464 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x464)
end = time.time()
print(end-start)
