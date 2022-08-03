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
        self.linear35 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout18 = Dropout(p=0.1, inplace=False)
        self.layernorm12 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x276, x274):
        x277=self.linear35(x276)
        x278=self.dropout18(x277)
        x279=operator.add(x278, x274)
        x280=self.layernorm12(x279)
        return x280

m = M().eval()
x276 = torch.randn(torch.Size([1, 384, 3072]))
x274 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x276, x274)
end = time.time()
print(end-start)
