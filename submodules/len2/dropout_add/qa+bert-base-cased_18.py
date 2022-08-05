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

    def forward(self, x403, x400):
        x404=self.dropout27(x403)
        x405=operator.add(x404, x400)
        return x405

m = M().eval()
x403 = torch.randn(torch.Size([1, 384, 768]))
x400 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x403, x400)
end = time.time()
print(end-start)