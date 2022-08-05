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
        self.linear6 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout3 = Dropout(p=0.1, inplace=False)
        self.layernorm2 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x67, x65):
        x68=self.linear6(x67)
        x69=self.dropout3(x68)
        x70=operator.add(x69, x65)
        x71=self.layernorm2(x70)
        return x71

m = M().eval()
x67 = torch.randn(torch.Size([1, 384, 1024]))
x65 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x67, x65)
end = time.time()
print(end-start)
