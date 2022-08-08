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
        self.dropout32 = Dropout(p=0.1, inplace=False)

    def forward(self, x481, x448):
        x482=self.dropout32(x481)
        x483=operator.add(x482, x448)
        return x483

m = M().eval()
x481 = torch.randn(torch.Size([1, 384, 768]))
x448 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x481, x448)
end = time.time()
print(end-start)
