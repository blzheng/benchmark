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
        self.dropout14 = Dropout(p=0.1, inplace=False)

    def forward(self, x230, x197):
        x231=self.dropout14(x230)
        x232=operator.add(x231, x197)
        return x232

m = M().eval()
x230 = torch.randn(torch.Size([1, 384, 256]))
x197 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x230, x197)
end = time.time()
print(end-start)
