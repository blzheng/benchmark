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
        self.linear44 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu22 = GELU(approximate='none')

    def forward(self, x264):
        x265=self.linear44(x264)
        x266=self.gelu22(x265)
        return x266

m = M().eval()
x264 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x264)
end = time.time()
print(end-start)
