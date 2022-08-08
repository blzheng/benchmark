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
        self.linear3 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x56, x59, x28):
        x60=x56.view(x59)
        x61=self.linear3(x60)
        x62=self.dropout2(x61)
        x63=operator.add(x62, x28)
        x64=self.layernorm1(x63)
        return x64

m = M().eval()
x56 = torch.randn(torch.Size([1, 384, 12, 64]))
x59 = (1, 384, 768, )
x28 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x56, x59, x28)
end = time.time()
print(end-start)
