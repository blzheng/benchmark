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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x279, x251):
        x280=self.linear4(x279)
        x281=self.dropout2(x280)
        x282=operator.add(x251, x281)
        x283=self.layernorm1(x282)
        return x283

m = M().eval()
x279 = torch.randn(torch.Size([1, 384, 768]))
x251 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x279, x251)
end = time.time()
print(end-start)
