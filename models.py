import torchvision.models as models
import torch
import transformers
import geffnet
import sys, os
from transformers import AutoModelForQuestionAnswering, AutoModelForTokenClassification, AutoModelForSequenceClassification, \
AutoModelForMultipleChoice, AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration

vision_models = ['alexnet', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'swin_t', 'swin_s', 'swin_b', \
                 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', \
                 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', \
                 'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inception_v3_google', 'googlenet', \
                 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', \
                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', \
                 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', \
                 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', \
                 'regnet_y_128gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_8gf','regnet_x_16gf', 'regnet_x_32gf', \
                 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14',
                ]
segmentation_models = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']
detection_models = ['fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn', \
                    'fcos_resnet50_fpn', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'retinanet_resnet50_fpn', \
                    'retinanet_resnet50_fpn_v2', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'keypointrcnn_resnet50_fpn' \
                   ]
video_models = ['r3d_18', 'mc3_18', 'r2plus1d_18', 'mvit_v1_b']
optical_flow_models = ['raft_large', 'raft_small']
geffnet_models = ['fbnetc_100', 'spnasnet_100', 'efficientnet_b0_geffnet', 'efficientnet_b1_geffnet', 'efficientnet_b2_geffnet', \
                  'efficientnet_b3_geffnet', 'efficientnet_b4_geffnet', 'efficientnet_b5_geffnet', 'efficientnet_b6_geffnet', 'efficientnet_b7_geffnet' \
                 ]
qa_models = ['qa+bert-base-cased', 'qa+albert-base-v1', 'qa+roberta-base', 'qa+xlm-roberta-base', \
             'qa+google/electra-base-generator', 'qa+google/electra-base-discriminator', 'qa+distilbert-base-cased', \
             'qa+xlnet-base-cased']
tokc_models = ['tokc+bert-base-cased', 'tokc+distilbert-base-cased', 'tokc+albert-base-v1', 'tokc+roberta-base',\
    'tokc+xlnet-base-cased', 'tokc+xlm-roberta-base', 'tokc+google/electra-base-generator', 'tokc+google/electra-base-discriminator']
txtc_models = ['txtc+bert-base-cased', 'txtc+distilbert-base-cased', 'txtc+albert-base-v1', 'txtc+roberta-base',\
    'txtc+xlnet-base-cased', 'txtc+xlm-roberta-base', 'txtc+google/electra-base-generator', 'txtc+google/electra-base-discriminator',\
    'txtc+allenai/longformer-base-4096', 'txtc+google/mobilebert-uncased', 'txtc+bert-base-chinese',\
    'txtc+distilbert-base-uncased-finetuned-sst-2-english', 'txtc+mrm8488/bert-tiny-finetuned-sms-spam-detection', \
    'txtc+microsoft/MiniLM-L12-H384-uncased']
mc_models = ['mc+bert-base-cased',	'mc+distilbert-base-cased', 'mc+albert-base-v1', \
	'mc+roberta-base', 'mc+xlnet-base-cased', 'mc+xlm-roberta-base', \
	'mc+google/electra-base-generator', 'mc+google/electra-base-discriminator']
mlm_models = ['mlm+bert-base-cased', 'mlm+distilbert-base-cased', 'mlm+albert-base-v1', \
	'mlm+roberta-base',	'mlm+xlm-roberta-base',	'mlm+google/electra-base-generator', \
    'mlm+google/electra-base-discriminator']
clm_models = ['clm+gpt2', 'clm+bert-base-cased', 'clm+roberta-base', 'clm+xlnet-base-cased', 'clm+xlm-roberta-base']
sum_models = ['sum+t5-small', 'sum+t5-base']

def get_model(name):
    if "+" in name:
        task, model = name.split("+")
        if task == "qa":
            return AutoModelForQuestionAnswering.from_pretrained(model)
        if task == "tokc":
            return AutoModelForTokenClassification.from_pretrained(model)
        if task == "txtc":
            return AutoModelForSequenceClassification.from_pretrained(model)
        if task == "mc":
            return AutoModelForMultipleChoice.from_pretrained(model)
        if task == "mlm":
            return AutoModelForMaskedLM.from_pretrained(model)
        if task == "clm":
            return AutoModelForCausalLM.from_pretrained(model)
        if task == "sum":
            return T5ForConditionalGeneration.from_pretrained(model)
    elif name in vision_models:
        return models.__dict__[name]()
    elif name in segmentation_models:
        return models.segmentation.__dict__[name]()
    elif name in detection_models:
        return models.detection.__dict__[name]()
    elif name in video_models:
        return models.video.__dict__[name]()
    elif name in optical_flow_models:
        return models.optical_flow.__dict__[name]()
    elif name in geffnet_models:
        return geffnet.create_model(name, pretrained=True)
    else:
        print("Model " + name + " is not supported.")
        exit()

def get_sample_inputs(m):
    batch_size = 1
    sequence_length = 384
    x = torch.ones((batch_size, sequence_length), dtype=torch.long)
    if isinstance(m, models.optical_flow.raft.RAFT):
        x = torch.rand(batch_size, 3, 440, 1024)
        input_dict = {'image1':x, 'image2':x, 'num_flow_updates': 32}
    elif isinstance(m, transformers.models.bert.modeling_bert.BertForQuestionAnswering) \
    or isinstance(m, transformers.models.albert.modeling_albert.AlbertForQuestionAnswering) \
    or isinstance(m, transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering) \
    or isinstance(m, transformers.models.electra.modeling_electra.ElectraForQuestionAnswering):
        input_dict = {'input_ids':x, 'attention_mask':x, 'token_type_ids':x, 'position_ids':None, \
        'head_mask':None, 'start_positions':None, 'inputs_embeds':None, 'end_positions': None, \
        'output_attentions': None, 'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.distilbert.modeling_distilbert.DistilBertForQuestionAnswering):
        input_dict = {'input_ids':x, 'attention_mask':x, 'head_mask':None, 'inputs_embeds':None, \
        'start_positions': None, 'end_positions': None, 'output_attentions': None, \
        'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.bert.modeling_bert.BertForTokenClassification) \
    or isinstance(m, transformers.models.albert.modeling_albert.AlbertForTokenClassification) \
    or isinstance(m, transformers.models.roberta.modeling_roberta.RobertaForTokenClassification) \
    or isinstance(m, transformers.models.electra.modeling_electra.ElectraForTokenClassification) \
    or isinstance(m, transformers.models.bert.modeling_bert.BertForSequenceClassification) \
    or isinstance(m, transformers.models.albert.modeling_albert.AlbertForSequenceClassification) \
    or isinstance(m, transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification) \
    or isinstance(m, transformers.models.electra.modeling_electra.ElectraForSequenceClassification) \
    or isinstance(m, transformers.models.mobilebert.modeling_mobilebert.MobileBertForSequenceClassification):
        input_dict =  {'input_ids':x, 'attention_mask':x, 'token_type_ids':x, 'position_ids':None, \
        'head_mask': None, 'inputs_embeds': None, 'labels': None, 'output_attentions': None, \
        'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.distilbert.modeling_distilbert.DistilBertForTokenClassification) \
    or isinstance(m, transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification) \
    or isinstance(m, transformers.models.distilbert.modeling_distilbert.DistilBertForMaskedLM):
        input_dict =  {'input_ids':x, 'attention_mask':x, 'head_mask': None, 'inputs_embeds': None, \
        'labels': None, 'output_attentions': None, 'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.distilbert.modeling_distilbert.DistilBertForMultipleChoice):
        x = torch.ones((batch_size, 4, sequence_length), dtype=torch.long)
        input_dict =  {'input_ids':x, 'attention_mask':x, 'head_mask': None, 'inputs_embeds': None, \
        'labels': None, 'output_attentions': None, 'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.bert.modeling_bert.BertForMultipleChoice) \
    or isinstance(m, transformers.models.albert.modeling_albert.AlbertForMultipleChoice) \
    or isinstance(m, transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice) \
    or isinstance(m, transformers.models.electra.modeling_electra.ElectraForMultipleChoice):
        x = torch.ones((batch_size, 4, sequence_length), dtype=torch.long)
        input_dict =  {'input_ids':x, 'attention_mask':x, 'token_type_ids':None, 'position_ids':None, \
        'head_mask': None, 'inputs_embeds': None, 'labels': None, 'output_attentions': None, \
        'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimple) \
    or isinstance(m, transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassification) \
    or isinstance(m, transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassification) \
    or isinstance(m, transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModel):
        input_dict =  {'input_ids':x, 'attention_mask':x, 'mems': None, 'perm_mask':None, \
        'target_mapping': None, 'token_type_ids':x, 'input_mask':None, \
        'head_mask': None, 'inputs_embeds': None, 'labels': None, 'use_mems': None, 'output_attentions': None, \
        'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoice):
        x = torch.ones((batch_size, 4, sequence_length), dtype=torch.long)
        input_dict =  {'input_ids':x, 'attention_mask':x, 'mems': None, 'perm_mask':None, \
        'target_mapping': None, 'token_type_ids':x, 'input_mask':None, \
        'head_mask': None, 'inputs_embeds': None, 'labels': None, 'use_mems': None, 'output_attentions': None, \
        'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.longformer.modeling_longformer.LongformerForSequenceClassification):
        input_dict =  {'input_ids':x, 'attention_mask':x, 'global_attention_mask': None, 'head_mask':None, \
        'token_type_ids':x, 'position_ids':None, 'inputs_embeds': None, 'labels': None, 'output_attentions': None, \
        'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.bert.modeling_bert.BertForMaskedLM) \
    or isinstance(m, transformers.models.albert.modeling_albert.AlbertForMaskedLM) \
    or isinstance(m, transformers.models.roberta.modeling_roberta.RobertaForMaskedLM) \
    or isinstance(m, transformers.models.electra.modeling_electra.ElectraForMaskedLM):
        input_dict =  {'input_ids':x, 'attention_mask':x, 'token_type_ids':x, 'position_ids': None, \
        'head_mask': None, 'inputs_embeds': None, 'encoder_hidden_states': None, 'encoder_attention_mask': None, \
        'labels': None, 'output_attentions': None, 'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.bert.modeling_bert.BertLMHeadModel) \
    or isinstance(m, transformers.models.roberta.modeling_roberta.RobertaForCausalLM) \
    or isinstance(m, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
        input_dict =  {'input_ids':x, 'attention_mask':x, 'token_type_ids':x, 'position_ids': None, \
        'head_mask': None, 'inputs_embeds': None, 'encoder_hidden_states': None, 'encoder_attention_mask': None, \
        'labels': None, 'past_key_values': None, 'use_cache': None, 'output_attentions': None, \
        'output_hidden_states': None, 'return_dict': None}
    elif isinstance(m, transformers.models.t5.modeling_t5.T5ForConditionalGeneration):
        x = torch.ones((batch_size, sequence_length), dtype=torch.long)
        y = torch.ones((batch_size, 1), dtype=torch.long)
        input_dict =  {'input_ids':x, 'attention_mask':x, 'decoder_input_ids':y, 'decoder_attention_mask': None, \
        'head_mask': None, 'decoder_head_mask': None, 'cross_attn_head_mask': None, 'encoder_outputs': None, \
        'past_key_values': None, 'inputs_embeds': None, 'decoder_inputs_embeds': None, 'labels':None, \
        'use_cache': None, 'output_attentions': None, 'output_hidden_states': None, 'return_dict': None}
    return input_dict

def get_inputs_dict(name, inputs):
    inputs_dict = {}
    if "+" in name:
        task, _ = name.split("+")
        if task == "txtc" or task == "qa" or task == "tokc" or task == "mlm" or task == "clm":
            for i in range(len(inputs)):
                if "input_ids" in inputs[i] \
                or ("attention_mask" in inputs[i] and not "_attention" in inputs[i]) \
                or "token_type_ids" in inputs[i]:
                    inputs_dict[inputs[i]] = "torch.ones((1, 384), dtype=torch.long)"
                else:
                    inputs_dict[inputs[i]] = "None"
        if task == "mc":
            for i in range(len(inputs)):
                if "input_ids" in inputs[i] \
                or ("attention_mask" in inputs[i] and not "_attention" in inputs[i]):
                    inputs_dict[inputs[i]] = "torch.ones((1, 4, 384), dtype=torch.long)"
                else:
                    inputs_dict[inputs[i]] = "None"
        if task == "sum":
            for i in range(len(inputs)):
                if  "input_ids" in inputs[i] \
                or ("attention_mask" in inputs[i] and not "_attention" in inputs[i]):
                    inputs_dict[inputs[i]] = "torch.ones((1, 384), dtype=torch.long)"
                elif decoder_input_ids in inputs[i]:
                    inputs_dict[inputs[i]] = "torch.ones((1, 1), dtype=torch.long)"
                else:
                    inputs_dict[inputs[i]] = "None"
    elif name in vision_models or name in geffnet_models or name in segmentation_models:
        inputs_dict[inputs[0]] = "torch.rand(1, 3, 224, 224)"
    elif name in optical_flow_models:
        inputs_dict[inputs[0]] = "torch.rand(1, 3, 440, 1024)"
        inputs_dict[inputs[1]] = "torch.rand(1, 3, 440, 1024)"
    return inputs_dict
