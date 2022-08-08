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
        self.dropout3 = Dropout(p=0.1, inplace=False)

    def forward(self, x68, x65):
        x69=self.dropout3(x68)
        x70=operator.add(x69, x65)
        return x70

m = M().eval()
x68 = torch.randn(torch.Size([1, 384, 256]))
x65 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x68, x65)
end = time.time()
print(end-start)
