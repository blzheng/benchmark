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
        self.conv2d80 = Conv2d(720, 720, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=720, bias=False)
        self.batchnorm2d48 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x246):
        x247=self.conv2d80(x246)
        x248=self.batchnorm2d48(x247)
        return x248

m = M().eval()
x246 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x246)
end = time.time()
print(end-start)
