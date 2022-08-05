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
        self.gelu13 = GELU(approximate='none')
        self.dropout26 = Dropout(p=0.0, inplace=False)

    def forward(self, x335):
        x336=self.gelu13(x335)
        x337=self.dropout26(x336)
        return x337

m = M().eval()
x335 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x335)
end = time.time()
print(end-start)
