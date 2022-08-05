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
        self.linear30 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout15 = Dropout(p=0.1, inplace=False)
        self.layernorm10 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear31 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x235, x233):
        x236=self.linear30(x235)
        x237=self.dropout15(x236)
        x238=operator.add(x237, x233)
        x239=self.layernorm10(x238)
        x240=self.linear31(x239)
        return x240

m = M().eval()
x235 = torch.randn(torch.Size([1, 384, 1024]))
x233 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x235, x233)
end = time.time()
print(end-start)
