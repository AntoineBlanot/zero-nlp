from collections import OrderedDict
from typing import Tuple

import math
import torch
from torch import Tensor, nn, FloatTensor, LongTensor, BoolTensor
from transformers import T5Model, RobertaModel
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.roberta.configuration_roberta import RobertaConfig


class T5ForClassification(T5Model):
    """
    Module based on `T5` models but adapted for `sequence classification` task
    """

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.classif_head = torch.nn.Sequential(OrderedDict([
            ('dropout', torch.nn.Dropout(config.dropout_rate)),
            ('out_proj', torch.nn.utils.skip_init(torch.nn.Linear, config.d_model, config.n_class, bias=True))
        ]))
        # Init weights manually since it results in NaN
        stdv = 1. / math.sqrt(config.d_model)
        torch.nn.init.uniform_(self.classif_head.out_proj.weight, -stdv, stdv)
        torch.nn.init.uniform_(self.classif_head.out_proj.bias, -stdv, stdv)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids: LongTensor = None, labels: LongTensor = None, attention_mask: FloatTensor = None, decoder_input_ids: LongTensor = None, decoder_attention_mask: BoolTensor = None, head_mask: FloatTensor = None, decoder_head_mask: FloatTensor = None, cross_attn_head_mask: Tensor = None, encoder_outputs: Tuple[Tuple[FloatTensor]] = None, past_key_values: Tuple[Tuple[FloatTensor]] = None, inputs_embeds: Tensor = None, decoder_inputs_embeds: Tensor = None, use_cache: bool = None, output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None) -> Tuple[FloatTensor]:        
        outputs =  super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
        last_hidden_state = outputs.last_hidden_state
        pooled_last_state = last_hidden_state.mean(-2)
        logits = self.classif_head(pooled_last_state)
        
        outputs_dict = dict(
            logits=logits,
            pooled_last_state=pooled_last_state,
            last_hidden_state=last_hidden_state
        )

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs_dict.update({'loss': loss})

        return outputs_dict

class OldT5ForClassification(T5Model):
    """
    Deprecated
    """
    
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.head = torch.nn.Linear(self.config.d_model, self.config.n_class, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids: LongTensor = None, labels: LongTensor = None, attention_mask: FloatTensor = None, decoder_input_ids: LongTensor = None, decoder_attention_mask: BoolTensor = None, head_mask: FloatTensor = None, decoder_head_mask: FloatTensor = None, cross_attn_head_mask: Tensor = None, encoder_outputs: Tuple[Tuple[FloatTensor]] = None, past_key_values: Tuple[Tuple[FloatTensor]] = None, inputs_embeds: Tensor = None, decoder_inputs_embeds: Tensor = None, use_cache: bool = None, output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None) -> Tuple[FloatTensor]:        
        outputs =  super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
        last_hidden_state = outputs.last_hidden_state
        logits = self.head(last_hidden_state[:, 0, :])
        
        outputs_dict = dict(
            logits=logits,
            last_hidden_state=last_hidden_state
        )

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs_dict.update({'loss': loss})

        return outputs_dict

class RobertaForClassification(RobertaModel):
    """
    Module based on `BERT` models but adapted for `sequence classification` task
    """

    RobertaModel._keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config: RobertaConfig, add_pooling_layer: bool = False):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.classif_head = torch.nn.Sequential(OrderedDict([
            ('dense', torch.nn.Linear(config.hidden_size, config.hidden_size)),
            ('dropout', torch.nn.Dropout(config.hidden_dropout_prob)),
            ('out_proj', torch.nn.Linear(config.hidden_size, config.n_class))
        ]))
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids: Tensor = None, labels: LongTensor = None, attention_mask: Tensor = None, token_type_ids: Tensor = None, position_ids: Tensor = None, inputs_embeds: Tensor = None, output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None) -> Tuple[FloatTensor]:
        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)
        last_hidden_state = outputs.last_hidden_state
        logits = self.classif_head(last_hidden_state[:, 0, :])

        outputs_dict = dict(
            logits=logits,
            last_hidden_state=last_hidden_state
        )

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs_dict.update({'loss': loss})

        return outputs_dict

class RobertaForTokenClassification(RobertaModel):
    """
    Module based on `BERT` models but adapted for `token classification` task
    """

    RobertaModel._keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config: RobertaConfig, add_pooling_layer: bool = False):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.token_head = torch.nn.Sequential(OrderedDict([
            ('dropout', torch.nn.Dropout(config.hidden_dropout_prob)),
            ('out_proj', torch.nn.Linear(config.hidden_size, config.n_class))
        ]))
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids: Tensor = None, labels: LongTensor = None, attention_mask: Tensor = None, token_type_ids: Tensor = None, position_ids: Tensor = None, inputs_embeds: Tensor = None, output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None) -> Tuple[FloatTensor]:
        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)
        last_hidden_state = outputs.last_hidden_state
        logits = self.token_head(last_hidden_state)

        outputs_dict = dict(
            logits=logits,
            last_hidden_state=last_hidden_state
        )

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.config.n_class), labels.view(-1))
            outputs_dict.update({'loss': loss})

        return outputs_dict
