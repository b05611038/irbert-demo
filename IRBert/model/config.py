# NOTE:
# This config file is a simplified version of the original IR-BERT configuration.
# Architecture details related to the spectral processor have been removed or replaced
# to protect unpublished innovations.

from ..utils import Configuration


__all__ = ['IRBertConfig']


class IRBertConfig(Configuration):
    def __init__(self,
            vocab_size = 3,
            hidden_size = 768,
            num_hidden_layer = 12,
            num_hidden_head = 12,
            feedforward_size = 3072,
            hidden_act = 'gelu',
            dropout_prob = 0.1,
            attn_mask_value = -1e9,
            max_position_embeddings = 4096,
            initializer_range = 0.02,
            layer_norm_eps = 1e-12,
            mask_token = '[MASK]',
            mask_token_id = 0,
            pad_token = '[PAD]',
            pad_token_id = 1,
            sep_token = '[SEP]',
            sep_token_id = 2,
            position_embedding_type = 'absolute',
            # ⛔ Sensitive wavelength embedding logic removed
            # ⛔ CNN processor architecture removed
            handled_tasks = None):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_head = num_hidden_head
        self.feedforward_size = feedforward_size
        self.hidden_act = hidden_act
        self.dropout_prob = dropout_prob
        self.attn_mask_value = attn_mask_value
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.sep_token = sep_token
        self.sep_token_id = sep_token_id
        self.position_embedding_type = position_embedding_type
        self.handled_tasks = handled_tasks

        super(IRBertConfig, self).__init__(
                vocab_size = vocab_size,
                hidden_size = hidden_size,
                num_hidden_layer = num_hidden_layer,
                num_hidden_head = num_hidden_head,
                feedforward_size = feedforward_size,
                hidden_act = hidden_act,
                dropout_prob = dropout_prob,
                attn_mask_value = attn_mask_value,
                max_position_embeddings = max_position_embeddings,
                initializer_range = initializer_range,
                layer_norm_eps = layer_norm_eps,
                mask_token = mask_token,
                mask_token_id = mask_token_id,
                pad_token = pad_token,
                pad_token_id = pad_token_id,
                sep_token = sep_token,
                sep_token_id = sep_token_id,
                position_embedding_type = position_embedding_type,
                handled_tasks = handled_tasks)



