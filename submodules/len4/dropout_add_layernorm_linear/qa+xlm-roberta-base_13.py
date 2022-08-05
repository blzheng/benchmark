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
        self.dropout20 = Dropout(p=0.1, inplace=False)
        self.layernorm13 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear40 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x313, x280):
        x314=self.dropout20(x313)
        x315=operator.add(x314, x280)
        x316=self.layernorm13(x315)
        x317=self.linear40(x316)
        return x317

m = M().eval()
x313 = torch.randn(torch.Size([1, 384, 768]))
x280 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x313, x280)
end = time.time()
print(end-start)
