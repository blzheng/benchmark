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
        self.dropout7 = Dropout(p=0.1, inplace=False)

    def forward(self, x137, x126):
        x138=self.dropout7(x137)
        x139=torch.matmul(x138, x126)
        return x139

m = M().eval()
x137 = torch.randn(torch.Size([1, 4, 384, 384]))
x126 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x137, x126)
end = time.time()
print(end-start)