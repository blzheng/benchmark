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
        self.linear25 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)

    def forward(self, x296):
        x297=self.linear25(x296)
        x298=self.gelu11(x297)
        x299=self.dropout22(x298)
        return x299

m = M().eval()
x296 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x296)
end = time.time()
print(end-start)
