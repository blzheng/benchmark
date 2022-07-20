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
        self.conv2d115 = Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
        self.batchnorm2d69 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x341):
        x342=self.conv2d115(x341)
        x343=self.batchnorm2d69(x342)
        return x343

m = M().eval()
x341 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x341)
end = time.time()
print(end-start)
