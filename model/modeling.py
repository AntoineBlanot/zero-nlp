from typing import Tuple

import torch
from torch import Tensor, nn, FloatTensor, LongTensor, BoolTensor
from transformers import T5Model
from transformers.models.t5.configuration_t5 import T5Config


class T5ForClassification(T5Model):
    
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