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
        self.dropout13 = Dropout(p=0.1, inplace=False)

    def forward(self, x221):
        x222=self.dropout13(x221)
        return x222

m = M().eval()
x221 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x221)
end = time.time()
print(end-start)
