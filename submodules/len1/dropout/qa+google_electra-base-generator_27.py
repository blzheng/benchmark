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
        self.dropout27 = Dropout(p=0.1, inplace=False)

    def forward(self, x404):
        x405=self.dropout27(x404)
        return x405

m = M().eval()
x404 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
