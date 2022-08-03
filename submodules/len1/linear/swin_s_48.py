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
        self.linear48 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x552):
        x553=self.linear48(x552)
        return x553

m = M().eval()
x552 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x552)
end = time.time()
print(end-start)
