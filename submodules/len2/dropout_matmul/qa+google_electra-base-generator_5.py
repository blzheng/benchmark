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
        self.dropout16 = Dropout(p=0.1, inplace=False)

    def forward(self, x263, x252):
        x264=self.dropout16(x263)
        x265=torch.matmul(x264, x252)
        return x265

m = M().eval()
x263 = torch.randn(torch.Size([1, 4, 384, 384]))
x252 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x263, x252)
end = time.time()
print(end-start)
