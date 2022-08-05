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
        self.layernorm7 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.linear5 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu2 = GELU(approximate='none')
        self.dropout4 = Dropout(p=0.0, inplace=False)

    def forward(self, x72):
        x73=self.layernorm7(x72)
        x74=self.linear5(x73)
        x75=self.gelu2(x74)
        x76=self.dropout4(x75)
        return x76

m = M().eval()
x72 = torch.randn(torch.Size([1, 28, 28, 192]))
start = time.time()
output = m(x72)
end = time.time()
print(end-start)
