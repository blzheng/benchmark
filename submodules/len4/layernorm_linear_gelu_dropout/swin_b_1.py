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
        self.layernorm4 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.linear2 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.dropout2 = Dropout(p=0.0, inplace=False)

    def forward(self, x41):
        x42=self.layernorm4(x41)
        x43=self.linear2(x42)
        x44=self.gelu1(x43)
        x45=self.dropout2(x44)
        return x45

m = M().eval()
x41 = torch.randn(torch.Size([1, 56, 56, 128]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)
