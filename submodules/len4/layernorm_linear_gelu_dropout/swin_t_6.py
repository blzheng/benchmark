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
        self.layernorm16 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear14 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)

    def forward(self, x172):
        x173=self.layernorm16(x172)
        x174=self.linear14(x173)
        x175=self.gelu6(x174)
        x176=self.dropout12(x175)
        return x176

m = M().eval()
x172 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x172)
end = time.time()
print(end-start)
