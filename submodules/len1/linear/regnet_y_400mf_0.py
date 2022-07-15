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
        self.linear0 = Linear(in_features=440, out_features=1000, bias=True)

    def forward(self, x269):
        x270=self.linear0(x269)
        return x270

m = M().eval()
x269 = torch.randn(torch.Size([1, 440]))
start = time.time()
output = m(x269)
end = time.time()
print(end-start)
