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
        self.linear28 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x225, x228):
        x229=x225.view(x228)
        x230=self.linear28(x229)
        return x230

m = M().eval()
x225 = torch.randn(torch.Size([1, 384, 4, 64]))
x228 = (1, 384, 256, )
start = time.time()
output = m(x225, x228)
end = time.time()
print(end-start)
