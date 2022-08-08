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

    def forward(self, x136, x125):
        x137=self.dropout7(x136)
        x138=torch.matmul(x137, x125)
        return x138

m = M().eval()
x136 = torch.randn(torch.Size([1, 12, 384, 384]))
x125 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x136, x125)
end = time.time()
print(end-start)
