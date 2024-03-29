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
        self.dropout14 = Dropout(p=0.1, inplace=False)
        self.layernorm9 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x229, x196):
        x230=self.dropout14(x229)
        x231=operator.add(x230, x196)
        x232=self.layernorm9(x231)
        return x232

m = M().eval()
x229 = torch.randn(torch.Size([1, 384, 768]))
x196 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x229, x196)
end = time.time()
print(end-start)
