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
        self.dropout9 = Dropout(p=0.0, inplace=False)

    def forward(self, x131):
        x132=self.dropout9(x131)
        return x132

m = M().eval()
x131 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)