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

    def forward(self, x401, x410):
        x411=x401.view(x410)
        return x411

m = M().eval()
x401 = torch.randn(torch.Size([1, 384, 768]))
x410 = (1, 384, 12, 64, )
start = time.time()
output = m(x401, x410)
end = time.time()
print(end-start)
