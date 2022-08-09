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
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.embedding0 = Embedding(50265, 768, padding_idx=1)
        self.embedding1 = Embedding(1, 768)
        self.embedding2 = Embedding(514, 768, padding_idx=1)
        self.layernorm0 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.dropout0 = Dropout(p=0.1, inplace=False)
        self.linear0 = Linear(in_features=768, out_features=768, bias=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)
        self.linear2 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout1 = Dropout(p=0.1, inplace=False)
        self.linear3 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear4 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear5 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout3 = Dropout(p=0.1, inplace=False)
        self.layernorm2 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear6 = Linear(in_features=768, out_features=768, bias=True)
        self.linear7 = Linear(in_features=768, out_features=768, bias=True)
        self.linear8 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout4 = Dropout(p=0.1, inplace=False)
        self.linear9 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout5 = Dropout(p=0.1, inplace=False)
        self.layernorm3 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear11 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout6 = Dropout(p=0.1, inplace=False)
        self.layernorm4 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=768, out_features=768, bias=True)
        self.linear13 = Linear(in_features=768, out_features=768, bias=True)
        self.linear14 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout7 = Dropout(p=0.1, inplace=False)
        self.linear15 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout8 = Dropout(p=0.1, inplace=False)
        self.layernorm5 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear17 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout9 = Dropout(p=0.1, inplace=False)
        self.layernorm6 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=768, out_features=768, bias=True)
        self.linear19 = Linear(in_features=768, out_features=768, bias=True)
        self.linear20 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout10 = Dropout(p=0.1, inplace=False)
        self.linear21 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout11 = Dropout(p=0.1, inplace=False)
        self.layernorm7 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear22 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear23 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout12 = Dropout(p=0.1, inplace=False)
        self.layernorm8 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear24 = Linear(in_features=768, out_features=768, bias=True)
        self.linear25 = Linear(in_features=768, out_features=768, bias=True)
        self.linear26 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout13 = Dropout(p=0.1, inplace=False)
        self.linear27 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout14 = Dropout(p=0.1, inplace=False)
        self.layernorm9 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear28 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear29 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout15 = Dropout(p=0.1, inplace=False)
        self.layernorm10 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear30 = Linear(in_features=768, out_features=768, bias=True)
        self.linear31 = Linear(in_features=768, out_features=768, bias=True)
        self.linear32 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout16 = Dropout(p=0.1, inplace=False)
        self.linear33 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout17 = Dropout(p=0.1, inplace=False)
        self.layernorm11 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear34 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear35 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout18 = Dropout(p=0.1, inplace=False)
        self.layernorm12 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=768, out_features=768, bias=True)
        self.linear37 = Linear(in_features=768, out_features=768, bias=True)
        self.linear38 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout19 = Dropout(p=0.1, inplace=False)
        self.linear39 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout20 = Dropout(p=0.1, inplace=False)
        self.layernorm13 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear40 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear41 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout21 = Dropout(p=0.1, inplace=False)
        self.layernorm14 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear42 = Linear(in_features=768, out_features=768, bias=True)
        self.linear43 = Linear(in_features=768, out_features=768, bias=True)
        self.linear44 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout22 = Dropout(p=0.1, inplace=False)
        self.linear45 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout23 = Dropout(p=0.1, inplace=False)
        self.layernorm15 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear46 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear47 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout24 = Dropout(p=0.1, inplace=False)
        self.layernorm16 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear48 = Linear(in_features=768, out_features=768, bias=True)
        self.linear49 = Linear(in_features=768, out_features=768, bias=True)
        self.linear50 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout25 = Dropout(p=0.1, inplace=False)
        self.linear51 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout26 = Dropout(p=0.1, inplace=False)
        self.layernorm17 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear52 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear53 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout27 = Dropout(p=0.1, inplace=False)
        self.layernorm18 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear54 = Linear(in_features=768, out_features=768, bias=True)
        self.linear55 = Linear(in_features=768, out_features=768, bias=True)
        self.linear56 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout28 = Dropout(p=0.1, inplace=False)
        self.linear57 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout29 = Dropout(p=0.1, inplace=False)
        self.layernorm19 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear58 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear59 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout30 = Dropout(p=0.1, inplace=False)
        self.layernorm20 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear60 = Linear(in_features=768, out_features=768, bias=True)
        self.linear61 = Linear(in_features=768, out_features=768, bias=True)
        self.linear62 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout31 = Dropout(p=0.1, inplace=False)
        self.linear63 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout32 = Dropout(p=0.1, inplace=False)
        self.layernorm21 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear64 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear65 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout33 = Dropout(p=0.1, inplace=False)
        self.layernorm22 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear66 = Linear(in_features=768, out_features=768, bias=True)
        self.linear67 = Linear(in_features=768, out_features=768, bias=True)
        self.linear68 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout34 = Dropout(p=0.1, inplace=False)
        self.linear69 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout35 = Dropout(p=0.1, inplace=False)
        self.layernorm23 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear70 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear71 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout36 = Dropout(p=0.1, inplace=False)
        self.layernorm24 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear72 = Linear(in_features=768, out_features=2, bias=True)
        self._tensor_constant00 = torch.rand(torch.Size([1, 384])).to(torch.int64)
        self._tensor_constant01 = torch.rand(torch.Size([1, 384])).to(torch.int64)
        self._tensor_constant10 = torch.rand(torch.Size([1, 384])).to(torch.int64)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant21 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant22 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant23 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant24 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant25 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant26 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant27 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant28 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant29 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant210 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)
        self._tensor_constant211 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, head_mask_1, inputs_embeds_1, start_positions_1, end_positions_1, output_attentions_1, output_hidden_states_1, return_dict_1):
        x0=input_ids_1
        x1=attention_mask_1
        x2=token_type_ids_1
        x3=position_ids_1
        x4=torch.fx._symbolic_trace._assert_is_none(x3, 'position_ids has been specialized to have value None but got another value')
        x5=head_mask_1
        x6=torch.fx._symbolic_trace._assert_is_none(x5, 'head_mask has been specialized to have value None but got another value')
        x7=inputs_embeds_1
        x8=torch.fx._symbolic_trace._assert_is_none(x7, 'inputs_embeds has been specialized to have value None but got another value')
        x9=start_positions_1
        x10=torch.fx._symbolic_trace._assert_is_none(x9, 'start_positions has been specialized to have value None but got another value')
        x11=end_positions_1
        x12=torch.fx._symbolic_trace._assert_is_none(x11, 'end_positions has been specialized to have value None but got another value')
        x13=output_attentions_1
        x14=torch.fx._symbolic_trace._assert_is_none(x13, 'output_attentions has been specialized to have value None but got another value')
        x15=output_hidden_states_1
        x16=torch.fx._symbolic_trace._assert_is_none(x15, 'output_hidden_states has been specialized to have value None but got another value')
        x17=return_dict_1
        x18=torch.fx._symbolic_trace._assert_is_none(x17, 'return_dict has been specialized to have value None but got another value')
        x20=self.embedding0(self._tensor_constant00)
        x22=self.embedding1(self._tensor_constant00)
        x23=operator.add(x20, x22)
        x25=self.embedding2(self._tensor_constant10)
        x26=operator.add(x23, x25)
        x27=self.layernorm0(x26)
        x28=self.dropout0(x27)
        x29=self.linear0(x28)
        x30=self.linear1(x28)
        x31=x30.size()
        x32=operator.getitem(x31, slice(None, -1, None))
        x33=operator.add(x32, (12, 64))
        x34=x30.view(x33)
        x35=x34.permute(0, 2, 1, 3)
        x36=self.linear2(x28)
        x37=x36.size()
        x38=operator.getitem(x37, slice(None, -1, None))
        x39=operator.add(x38, (12, 64))
        x40=x36.view(x39)
        x41=x40.permute(0, 2, 1, 3)
        x42=x29.size()
        x43=operator.getitem(x42, slice(None, -1, None))
        x44=operator.add(x43, (12, 64))
        x45=x29.view(x44)
        x46=x45.permute(0, 2, 1, 3)
        x47=x35.transpose(-1, -2)
        x48=torch.matmul(x46, x47)
        x49=operator.truediv(x48, 8.0)
        x51=operator.add(x49, self._tensor_constant20)
        x52=torch.nn.functional.softmax(x51,dim=-1, _stacklevel=3, dtype=None)
        x53=self.dropout1(x52)
        x54=torch.matmul(x53, x41)
        x55=x54.permute(0, 2, 1, 3)
        x56=x55.contiguous()
        x57=x56.size()
        x58=operator.getitem(x57, slice(None, -2, None))
        x59=operator.add(x58, (768,))
        x60=x56.view(x59)
        x61=self.linear3(x60)
        x62=self.dropout2(x61)
        x63=operator.add(x62, x28)
        x64=self.layernorm1(x63)
        x65=self.linear4(x64)
        x66=torch._C._nn.gelu(x65)
        x67=self.linear5(x66)
        x68=self.dropout3(x67)
        x69=operator.add(x68, x64)
        x70=self.layernorm2(x69)
        x71=self.linear6(x70)
        x72=self.linear7(x70)
        x73=x72.size()
        x74=operator.getitem(x73, slice(None, -1, None))
        x75=operator.add(x74, (12, 64))
        x76=x72.view(x75)
        x77=x76.permute(0, 2, 1, 3)
        x78=self.linear8(x70)
        x79=x78.size()
        x80=operator.getitem(x79, slice(None, -1, None))
        x81=operator.add(x80, (12, 64))
        x82=x78.view(x81)
        x83=x82.permute(0, 2, 1, 3)
        x84=x71.size()
        x85=operator.getitem(x84, slice(None, -1, None))
        x86=operator.add(x85, (12, 64))
        x87=x71.view(x86)
        x88=x87.permute(0, 2, 1, 3)
        x89=x77.transpose(-1, -2)
        x90=torch.matmul(x88, x89)
        x91=operator.truediv(x90, 8.0)
        x93=operator.add(x91, self._tensor_constant20)
        x94=torch.nn.functional.softmax(x93,dim=-1, _stacklevel=3, dtype=None)
        x95=self.dropout4(x94)
        x96=torch.matmul(x95, x83)
        x97=x96.permute(0, 2, 1, 3)
        x98=x97.contiguous()
        x99=x98.size()
        x100=operator.getitem(x99, slice(None, -2, None))
        x101=operator.add(x100, (768,))
        x102=x98.view(x101)
        x103=self.linear9(x102)
        x104=self.dropout5(x103)
        x105=operator.add(x104, x70)
        x106=self.layernorm3(x105)
        x107=self.linear10(x106)
        x108=torch._C._nn.gelu(x107)
        x109=self.linear11(x108)
        x110=self.dropout6(x109)
        x111=operator.add(x110, x106)
        x112=self.layernorm4(x111)
        x113=self.linear12(x112)
        x114=self.linear13(x112)
        x115=x114.size()
        x116=operator.getitem(x115, slice(None, -1, None))
        x117=operator.add(x116, (12, 64))
        x118=x114.view(x117)
        x119=x118.permute(0, 2, 1, 3)
        x120=self.linear14(x112)
        x121=x120.size()
        x122=operator.getitem(x121, slice(None, -1, None))
        x123=operator.add(x122, (12, 64))
        x124=x120.view(x123)
        x125=x124.permute(0, 2, 1, 3)
        x126=x113.size()
        x127=operator.getitem(x126, slice(None, -1, None))
        x128=operator.add(x127, (12, 64))
        x129=x113.view(x128)
        x130=x129.permute(0, 2, 1, 3)
        x131=x119.transpose(-1, -2)
        x132=torch.matmul(x130, x131)
        x133=operator.truediv(x132, 8.0)
        x135=operator.add(x133, self._tensor_constant20)
        x136=torch.nn.functional.softmax(x135,dim=-1, _stacklevel=3, dtype=None)
        x137=self.dropout7(x136)
        x138=torch.matmul(x137, x125)
        x139=x138.permute(0, 2, 1, 3)
        x140=x139.contiguous()
        x141=x140.size()
        x142=operator.getitem(x141, slice(None, -2, None))
        x143=operator.add(x142, (768,))
        x144=x140.view(x143)
        x145=self.linear15(x144)
        x146=self.dropout8(x145)
        x147=operator.add(x146, x112)
        x148=self.layernorm5(x147)
        x149=self.linear16(x148)
        x150=torch._C._nn.gelu(x149)
        x151=self.linear17(x150)
        x152=self.dropout9(x151)
        x153=operator.add(x152, x148)
        x154=self.layernorm6(x153)
        x155=self.linear18(x154)
        x156=self.linear19(x154)
        x157=x156.size()
        x158=operator.getitem(x157, slice(None, -1, None))
        x159=operator.add(x158, (12, 64))
        x160=x156.view(x159)
        x161=x160.permute(0, 2, 1, 3)
        x162=self.linear20(x154)
        x163=x162.size()
        x164=operator.getitem(x163, slice(None, -1, None))
        x165=operator.add(x164, (12, 64))
        x166=x162.view(x165)
        x167=x166.permute(0, 2, 1, 3)
        x168=x155.size()
        x169=operator.getitem(x168, slice(None, -1, None))
        x170=operator.add(x169, (12, 64))
        x171=x155.view(x170)
        x172=x171.permute(0, 2, 1, 3)
        x173=x161.transpose(-1, -2)
        x174=torch.matmul(x172, x173)
        x175=operator.truediv(x174, 8.0)
        x177=operator.add(x175, self._tensor_constant20)
        x178=torch.nn.functional.softmax(x177,dim=-1, _stacklevel=3, dtype=None)
        x179=self.dropout10(x178)
        x180=torch.matmul(x179, x167)
        x181=x180.permute(0, 2, 1, 3)
        x182=x181.contiguous()
        x183=x182.size()
        x184=operator.getitem(x183, slice(None, -2, None))
        x185=operator.add(x184, (768,))
        x186=x182.view(x185)
        x187=self.linear21(x186)
        x188=self.dropout11(x187)
        x189=operator.add(x188, x154)
        x190=self.layernorm7(x189)
        x191=self.linear22(x190)
        x192=torch._C._nn.gelu(x191)
        x193=self.linear23(x192)
        x194=self.dropout12(x193)
        x195=operator.add(x194, x190)
        x196=self.layernorm8(x195)
        x197=self.linear24(x196)
        x198=self.linear25(x196)
        x199=x198.size()
        x200=operator.getitem(x199, slice(None, -1, None))
        x201=operator.add(x200, (12, 64))
        x202=x198.view(x201)
        x203=x202.permute(0, 2, 1, 3)
        x204=self.linear26(x196)
        x205=x204.size()
        x206=operator.getitem(x205, slice(None, -1, None))
        x207=operator.add(x206, (12, 64))
        x208=x204.view(x207)
        x209=x208.permute(0, 2, 1, 3)
        x210=x197.size()
        x211=operator.getitem(x210, slice(None, -1, None))
        x212=operator.add(x211, (12, 64))
        x213=x197.view(x212)
        x214=x213.permute(0, 2, 1, 3)
        x215=x203.transpose(-1, -2)
        x216=torch.matmul(x214, x215)
        x217=operator.truediv(x216, 8.0)
        x219=operator.add(x217, self._tensor_constant20)
        x220=torch.nn.functional.softmax(x219,dim=-1, _stacklevel=3, dtype=None)
        x221=self.dropout13(x220)
        x222=torch.matmul(x221, x209)
        x223=x222.permute(0, 2, 1, 3)
        x224=x223.contiguous()
        x225=x224.size()
        x226=operator.getitem(x225, slice(None, -2, None))
        x227=operator.add(x226, (768,))
        x228=x224.view(x227)
        x229=self.linear27(x228)
        x230=self.dropout14(x229)
        x231=operator.add(x230, x196)
        x232=self.layernorm9(x231)
        x233=self.linear28(x232)
        x234=torch._C._nn.gelu(x233)
        x235=self.linear29(x234)
        x236=self.dropout15(x235)
        x237=operator.add(x236, x232)
        x238=self.layernorm10(x237)
        x239=self.linear30(x238)
        x240=self.linear31(x238)
        x241=x240.size()
        x242=operator.getitem(x241, slice(None, -1, None))
        x243=operator.add(x242, (12, 64))
        x244=x240.view(x243)
        x245=x244.permute(0, 2, 1, 3)
        x246=self.linear32(x238)
        x247=x246.size()
        x248=operator.getitem(x247, slice(None, -1, None))
        x249=operator.add(x248, (12, 64))
        x250=x246.view(x249)
        x251=x250.permute(0, 2, 1, 3)
        x252=x239.size()
        x253=operator.getitem(x252, slice(None, -1, None))
        x254=operator.add(x253, (12, 64))
        x255=x239.view(x254)
        x256=x255.permute(0, 2, 1, 3)
        x257=x245.transpose(-1, -2)
        x258=torch.matmul(x256, x257)
        x259=operator.truediv(x258, 8.0)
        x261=operator.add(x259, self._tensor_constant20)
        x262=torch.nn.functional.softmax(x261,dim=-1, _stacklevel=3, dtype=None)
        x263=self.dropout16(x262)
        x264=torch.matmul(x263, x251)
        x265=x264.permute(0, 2, 1, 3)
        x266=x265.contiguous()
        x267=x266.size()
        x268=operator.getitem(x267, slice(None, -2, None))
        x269=operator.add(x268, (768,))
        x270=x266.view(x269)
        x271=self.linear33(x270)
        x272=self.dropout17(x271)
        x273=operator.add(x272, x238)
        x274=self.layernorm11(x273)
        x275=self.linear34(x274)
        x276=torch._C._nn.gelu(x275)
        x277=self.linear35(x276)
        x278=self.dropout18(x277)
        x279=operator.add(x278, x274)
        x280=self.layernorm12(x279)
        x281=self.linear36(x280)
        x282=self.linear37(x280)
        x283=x282.size()
        x284=operator.getitem(x283, slice(None, -1, None))
        x285=operator.add(x284, (12, 64))
        x286=x282.view(x285)
        x287=x286.permute(0, 2, 1, 3)
        x288=self.linear38(x280)
        x289=x288.size()
        x290=operator.getitem(x289, slice(None, -1, None))
        x291=operator.add(x290, (12, 64))
        x292=x288.view(x291)
        x293=x292.permute(0, 2, 1, 3)
        x294=x281.size()
        x295=operator.getitem(x294, slice(None, -1, None))
        x296=operator.add(x295, (12, 64))
        x297=x281.view(x296)
        x298=x297.permute(0, 2, 1, 3)
        x299=x287.transpose(-1, -2)
        x300=torch.matmul(x298, x299)
        x301=operator.truediv(x300, 8.0)
        x303=operator.add(x301, self._tensor_constant20)
        x304=torch.nn.functional.softmax(x303,dim=-1, _stacklevel=3, dtype=None)
        x305=self.dropout19(x304)
        x306=torch.matmul(x305, x293)
        x307=x306.permute(0, 2, 1, 3)
        x308=x307.contiguous()
        x309=x308.size()
        x310=operator.getitem(x309, slice(None, -2, None))
        x311=operator.add(x310, (768,))
        x312=x308.view(x311)
        x313=self.linear39(x312)
        x314=self.dropout20(x313)
        x315=operator.add(x314, x280)
        x316=self.layernorm13(x315)
        x317=self.linear40(x316)
        x318=torch._C._nn.gelu(x317)
        x319=self.linear41(x318)
        x320=self.dropout21(x319)
        x321=operator.add(x320, x316)
        x322=self.layernorm14(x321)
        x323=self.linear42(x322)
        x324=self.linear43(x322)
        x325=x324.size()
        x326=operator.getitem(x325, slice(None, -1, None))
        x327=operator.add(x326, (12, 64))
        x328=x324.view(x327)
        x329=x328.permute(0, 2, 1, 3)
        x330=self.linear44(x322)
        x331=x330.size()
        x332=operator.getitem(x331, slice(None, -1, None))
        x333=operator.add(x332, (12, 64))
        x334=x330.view(x333)
        x335=x334.permute(0, 2, 1, 3)
        x336=x323.size()
        x337=operator.getitem(x336, slice(None, -1, None))
        x338=operator.add(x337, (12, 64))
        x339=x323.view(x338)
        x340=x339.permute(0, 2, 1, 3)
        x341=x329.transpose(-1, -2)
        x342=torch.matmul(x340, x341)
        x343=operator.truediv(x342, 8.0)
        x345=operator.add(x343, self._tensor_constant20)
        x346=torch.nn.functional.softmax(x345,dim=-1, _stacklevel=3, dtype=None)
        x347=self.dropout22(x346)
        x348=torch.matmul(x347, x335)
        x349=x348.permute(0, 2, 1, 3)
        x350=x349.contiguous()
        x351=x350.size()
        x352=operator.getitem(x351, slice(None, -2, None))
        x353=operator.add(x352, (768,))
        x354=x350.view(x353)
        x355=self.linear45(x354)
        x356=self.dropout23(x355)
        x357=operator.add(x356, x322)
        x358=self.layernorm15(x357)
        x359=self.linear46(x358)
        x360=torch._C._nn.gelu(x359)
        x361=self.linear47(x360)
        x362=self.dropout24(x361)
        x363=operator.add(x362, x358)
        x364=self.layernorm16(x363)
        x365=self.linear48(x364)
        x366=self.linear49(x364)
        x367=x366.size()
        x368=operator.getitem(x367, slice(None, -1, None))
        x369=operator.add(x368, (12, 64))
        x370=x366.view(x369)
        x371=x370.permute(0, 2, 1, 3)
        x372=self.linear50(x364)
        x373=x372.size()
        x374=operator.getitem(x373, slice(None, -1, None))
        x375=operator.add(x374, (12, 64))
        x376=x372.view(x375)
        x377=x376.permute(0, 2, 1, 3)
        x378=x365.size()
        x379=operator.getitem(x378, slice(None, -1, None))
        x380=operator.add(x379, (12, 64))
        x381=x365.view(x380)
        x382=x381.permute(0, 2, 1, 3)
        x383=x371.transpose(-1, -2)
        x384=torch.matmul(x382, x383)
        x385=operator.truediv(x384, 8.0)
        x387=operator.add(x385, self._tensor_constant20)
        x388=torch.nn.functional.softmax(x387,dim=-1, _stacklevel=3, dtype=None)
        x389=self.dropout25(x388)
        x390=torch.matmul(x389, x377)
        x391=x390.permute(0, 2, 1, 3)
        x392=x391.contiguous()
        x393=x392.size()
        x394=operator.getitem(x393, slice(None, -2, None))
        x395=operator.add(x394, (768,))
        x396=x392.view(x395)
        x397=self.linear51(x396)
        x398=self.dropout26(x397)
        x399=operator.add(x398, x364)
        x400=self.layernorm17(x399)
        x401=self.linear52(x400)
        x402=torch._C._nn.gelu(x401)
        x403=self.linear53(x402)
        x404=self.dropout27(x403)
        x405=operator.add(x404, x400)
        x406=self.layernorm18(x405)
        x407=self.linear54(x406)
        x408=self.linear55(x406)
        x409=x408.size()
        x410=operator.getitem(x409, slice(None, -1, None))
        x411=operator.add(x410, (12, 64))
        x412=x408.view(x411)
        x413=x412.permute(0, 2, 1, 3)
        x414=self.linear56(x406)
        x415=x414.size()
        x416=operator.getitem(x415, slice(None, -1, None))
        x417=operator.add(x416, (12, 64))
        x418=x414.view(x417)
        x419=x418.permute(0, 2, 1, 3)
        x420=x407.size()
        x421=operator.getitem(x420, slice(None, -1, None))
        x422=operator.add(x421, (12, 64))
        x423=x407.view(x422)
        x424=x423.permute(0, 2, 1, 3)
        x425=x413.transpose(-1, -2)
        x426=torch.matmul(x424, x425)
        x427=operator.truediv(x426, 8.0)
        x429=operator.add(x427, self._tensor_constant20)
        x430=torch.nn.functional.softmax(x429,dim=-1, _stacklevel=3, dtype=None)
        x431=self.dropout28(x430)
        x432=torch.matmul(x431, x419)
        x433=x432.permute(0, 2, 1, 3)
        x434=x433.contiguous()
        x435=x434.size()
        x436=operator.getitem(x435, slice(None, -2, None))
        x437=operator.add(x436, (768,))
        x438=x434.view(x437)
        x439=self.linear57(x438)
        x440=self.dropout29(x439)
        x441=operator.add(x440, x406)
        x442=self.layernorm19(x441)
        x443=self.linear58(x442)
        x444=torch._C._nn.gelu(x443)
        x445=self.linear59(x444)
        x446=self.dropout30(x445)
        x447=operator.add(x446, x442)
        x448=self.layernorm20(x447)
        x449=self.linear60(x448)
        x450=self.linear61(x448)
        x451=x450.size()
        x452=operator.getitem(x451, slice(None, -1, None))
        x453=operator.add(x452, (12, 64))
        x454=x450.view(x453)
        x455=x454.permute(0, 2, 1, 3)
        x456=self.linear62(x448)
        x457=x456.size()
        x458=operator.getitem(x457, slice(None, -1, None))
        x459=operator.add(x458, (12, 64))
        x460=x456.view(x459)
        x461=x460.permute(0, 2, 1, 3)
        x462=x449.size()
        x463=operator.getitem(x462, slice(None, -1, None))
        x464=operator.add(x463, (12, 64))
        x465=x449.view(x464)
        x466=x465.permute(0, 2, 1, 3)
        x467=x455.transpose(-1, -2)
        x468=torch.matmul(x466, x467)
        x469=operator.truediv(x468, 8.0)
        x471=operator.add(x469, self._tensor_constant20)
        x472=torch.nn.functional.softmax(x471,dim=-1, _stacklevel=3, dtype=None)
        x473=self.dropout31(x472)
        x474=torch.matmul(x473, x461)
        x475=x474.permute(0, 2, 1, 3)
        x476=x475.contiguous()
        x477=x476.size()
        x478=operator.getitem(x477, slice(None, -2, None))
        x479=operator.add(x478, (768,))
        x480=x476.view(x479)
        x481=self.linear63(x480)
        x482=self.dropout32(x481)
        x483=operator.add(x482, x448)
        x484=self.layernorm21(x483)
        x485=self.linear64(x484)
        x486=torch._C._nn.gelu(x485)
        x487=self.linear65(x486)
        x488=self.dropout33(x487)
        x489=operator.add(x488, x484)
        x490=self.layernorm22(x489)
        x491=self.linear66(x490)
        x492=self.linear67(x490)
        x493=x492.size()
        x494=operator.getitem(x493, slice(None, -1, None))
        x495=operator.add(x494, (12, 64))
        x496=x492.view(x495)
        x497=x496.permute(0, 2, 1, 3)
        x498=self.linear68(x490)
        x499=x498.size()
        x500=operator.getitem(x499, slice(None, -1, None))
        x501=operator.add(x500, (12, 64))
        x502=x498.view(x501)
        x503=x502.permute(0, 2, 1, 3)
        x504=x491.size()
        x505=operator.getitem(x504, slice(None, -1, None))
        x506=operator.add(x505, (12, 64))
        x507=x491.view(x506)
        x508=x507.permute(0, 2, 1, 3)
        x509=x497.transpose(-1, -2)
        x510=torch.matmul(x508, x509)
        x511=operator.truediv(x510, 8.0)
        x513=operator.add(x511, self._tensor_constant20)
        x514=torch.nn.functional.softmax(x513,dim=-1, _stacklevel=3, dtype=None)
        x515=self.dropout34(x514)
        x516=torch.matmul(x515, x503)
        x517=x516.permute(0, 2, 1, 3)
        x518=x517.contiguous()
        x519=x518.size()
        x520=operator.getitem(x519, slice(None, -2, None))
        x521=operator.add(x520, (768,))
        x522=x518.view(x521)
        x523=self.linear69(x522)
        x524=self.dropout35(x523)
        x525=operator.add(x524, x490)
        x526=self.layernorm23(x525)
        x527=self.linear70(x526)
        x528=torch._C._nn.gelu(x527)
        x529=self.linear71(x528)
        x530=self.dropout36(x529)
        x531=operator.add(x530, x526)
        x532=self.layernorm24(x531)
        x533=self.linear72(x532)
        x534=x533.split(1,dim=-1)
        x535=operator.getitem(x534, 0)
        x536=operator.getitem(x534, 1)
        x537=x535.squeeze(-1)
        x538=x537.contiguous()
        x539=x536.squeeze(-1)
        x540=x539.contiguous()

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
input_ids_1 = torch.ones((1, 384), dtype=torch.long)
attention_mask_1 = torch.ones((1, 384), dtype=torch.long)
token_type_ids_1 = torch.ones((1, 384), dtype=torch.long)
position_ids_1 = None
head_mask_1 = None
inputs_embeds_1 = None
start_positions_1 = None
end_positions_1 = None
output_attentions_1 = None
output_hidden_states_1 = None
return_dict_1 = None
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, head_mask_1, inputs_embeds_1, start_positions_1, end_positions_1, output_attentions_1, output_hidden_states_1, return_dict_1)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
