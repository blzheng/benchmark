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
        self.layernorm9 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.linear7 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x80, x94):
        x95=operator.add(x80, x94)
        x96=self.layernorm9(x95)
        x97=self.linear7(x96)
        return x97

m = M().eval()
x80 = torch.randn(torch.Size([1, 28, 28, 256]))
x94 = torch.randn(torch.Size([1, 28, 28, 256]))
start = time.time()
output = m(x80, x94)
end = time.time()
print(end-start)
