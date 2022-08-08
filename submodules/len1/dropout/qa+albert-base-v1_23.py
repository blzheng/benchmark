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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x460):
        x461=self.dropout1(x460)
        return x461

m = M().eval()
x460 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x460)
end = time.time()
print(end-start)
