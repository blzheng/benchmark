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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x131, x103):
        x132=self.linear4(x131)
        x133=self.dropout2(x132)
        x134=operator.add(x103, x133)
        x135=self.layernorm1(x134)
        x136=self.linear5(x135)
        return x136

m = M().eval()
x131 = torch.randn(torch.Size([1, 384, 768]))
x103 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x131, x103)
end = time.time()
print(end-start)
