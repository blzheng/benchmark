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
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x280, x251):
        x281=self.dropout2(x280)
        x282=operator.add(x251, x281)
        return x282

m = M().eval()
x280 = torch.randn(torch.Size([1, 384, 768]))
x251 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x280, x251)
end = time.time()
print(end-start)
