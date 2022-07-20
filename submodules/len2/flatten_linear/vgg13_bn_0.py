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
        self.linear0 = Linear(in_features=25088, out_features=4096, bias=True)

    def forward(self, x36):
        x37=torch.flatten(x36, 1)
        x38=self.linear0(x37)
        return x38

m = M().eval()
x36 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x36)
end = time.time()
print(end-start)
