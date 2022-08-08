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
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x169, x140):
        x170=self.dropout2(x169)
        x171=operator.add(x140, x170)
        x172=self.layernorm1(x171)
        return x172

m = M().eval()
x169 = torch.randn(torch.Size([1, 384, 768]))
x140 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x169, x140)
end = time.time()
print(end-start)
