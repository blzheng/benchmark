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
        self.dropout36 = Dropout(p=0.1, inplace=False)

    def forward(self, x529, x526):
        x530=self.dropout36(x529)
        x531=operator.add(x530, x526)
        return x531

m = M().eval()
x529 = torch.randn(torch.Size([1, 384, 768]))
x526 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x529, x526)
end = time.time()
print(end-start)
