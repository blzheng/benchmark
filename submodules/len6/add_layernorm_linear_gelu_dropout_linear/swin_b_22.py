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
        self.layernorm49 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear47 = Linear(in_features=1024, out_features=4096, bias=True)
        self.gelu22 = GELU(approximate='none')
        self.dropout44 = Dropout(p=0.0, inplace=False)
        self.linear48 = Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, x533, x547):
        x548=operator.add(x533, x547)
        x549=self.layernorm49(x548)
        x550=self.linear47(x549)
        x551=self.gelu22(x550)
        x552=self.dropout44(x551)
        x553=self.linear48(x552)
        return x553

m = M().eval()
x533 = torch.randn(torch.Size([1, 7, 7, 1024]))
x547 = torch.randn(torch.Size([1, 7, 7, 1024]))
start = time.time()
output = m(x533, x547)
end = time.time()
print(end-start)
