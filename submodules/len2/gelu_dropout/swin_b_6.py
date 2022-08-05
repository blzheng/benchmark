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
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)

    def forward(self, x174):
        x175=self.gelu6(x174)
        x176=self.dropout12(x175)
        return x176

m = M().eval()
x174 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
