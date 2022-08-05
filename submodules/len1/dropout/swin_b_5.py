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
        self.dropout5 = Dropout(p=0.0, inplace=False)

    def forward(self, x77):
        x78=self.dropout5(x77)
        return x78

m = M().eval()
x77 = torch.randn(torch.Size([1, 28, 28, 256]))
start = time.time()
output = m(x77)
end = time.time()
print(end-start)
