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
        self.layernorm9 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.linear7 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu3 = GELU(approximate='none')
        self.dropout6 = Dropout(p=0.0, inplace=False)

    def forward(self, x95):
        x96=self.layernorm9(x95)
        x97=self.linear7(x96)
        x98=self.gelu3(x97)
        x99=self.dropout6(x98)
        return x99

m = M().eval()
x95 = torch.randn(torch.Size([1, 28, 28, 192]))
start = time.time()
output = m(x95)
end = time.time()
print(end-start)
