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
        self.batchnorm2d67 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)

    def forward(self, x346):
        x347=self.batchnorm2d67(x346)
        x348=self.relu85(x347)
        return x348

m = M().eval()
x346 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x346)
end = time.time()
print(end-start)
