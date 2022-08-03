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
        self.linear59 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout30 = Dropout(p=0.1, inplace=False)

    def forward(self, x444, x442):
        x445=self.linear59(x444)
        x446=self.dropout30(x445)
        x447=operator.add(x446, x442)
        return x447

m = M().eval()
x444 = torch.randn(torch.Size([1, 384, 3072]))
x442 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x444, x442)
end = time.time()
print(end-start)
