import torch


__all__ = ['BaseModelOutput', 'IRBertProcessorOutput', 'IRBertModelOutput', 
        'IRBertForPreTrainingOutput', 'IRBertForMaskedSMOutput', 'IRBertForMultiTaskPredictionOutput']


class BaseModelOutput:
    def __init__(self, **kwargs):
        self._data = kwargs
        self._exception_keys = ()
        self._must_existed_keys = tuple(kwargs.keys())

    @property
    def must_existed_keys(self):
        return self._must_existed_keys

    def __repr__(self):
        return '{0}(keys={1})'.format(self.__class__.__name__, 
                self.must_existed_keys)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if key not in self._exception_keys:
             assert key in self._data.keys()
             assert isinstance(value, torch.Tensor)

        self._data[key] = value
        return None

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()


class IRBertProcessorOutput(BaseModelOutput):
    pass  # fields inherited — no architecture leaked

class IRBertModelOutput(BaseModelOutput):
    def __init__(self, 
            combined_embeddings = None,
            last_hidden_state = None,
            last_attention = None,
            hidden_states = None,
            attentions = None):

        self.combined_embeddings = combined_embeddings
        self.last_hidden_state = last_hidden_state
        self.last_attention = last_attention
        self.hidden_states = hidden_states
        self.attentions = attentions

        super(IRBertModelOutput, self).__init__(
                combined_embeddings = combined_embeddings,
                last_hidden_state = last_hidden_state,
                last_attention = last_attention,
                hidden_states = hidden_states,
                attentions = attentions)


class IRBertForPreTrainingOutput(BaseModelOutput):
    def __init__(self,
            combined_embeddings = None,
            last_hidden_state = None,
            last_attention = None,
            hidden_states = None,
            attentions = None):

        self.combined_embeddings = combined_embeddings
        self.last_hidden_state = last_hidden_state
        self.last_attention = last_attention
        self.hidden_states = hidden_states
        self.attentions = attentions

        super(IRBertForPreTrainingOutput, self).__init__(
                combined_embeddings = combined_embeddings,
                last_hidden_state = last_hidden_state,
                last_attention = last_attention,
                hidden_states = hidden_states,
                attentions = attentions)


class IRBertForMaskedSMOutput(BaseModelOutput):
    def __init__(self,
            combined_embeddings = None,
            last_hidden_state = None,
            last_attention = None,
            hidden_states = None,
            attentions = None,
            reconstructed_spectrum = None,
            reconstructed_masked_spectrum = None):

        self.combined_embeddings = combined_embeddings
        self.last_hidden_state = last_hidden_state
        self.last_attention = last_attention
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.reconstructed_spectrum = reconstructed_spectrum
        self.reconstructed_masked_spectrum = reconstructed_masked_spectrum

        super(IRBertForMaskedSMOutput, self).__init__(
                combined_embeddings = combined_embeddings,
                last_hidden_state = last_hidden_state,
                last_attention = last_attention,
                hidden_states = hidden_states,
                attentions = attentions,
                reconstructed_spectrum = reconstructed_spectrum,
                reconstructed_masked_spectrum = reconstructed_masked_spectrum)


class IRBertForMultiTaskPredictionOutput(BaseModelOutput):
    def __init__(self,
            combined_embeddings = None,
            last_hidden_state = None,
            last_attention = None,
            hidden_states = None,
            attentions = None,
            tasks_prediction = None,
            tasks_weighting = None):

        self.combined_embeddings = combined_embeddings
        self.last_hidden_state = last_hidden_state
        self.last_attention = last_attention
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.tasks_prediction = tasks_prediction
        self.tasks_weighting = tasks_weighting

        super(IRBertForMultiTaskPredictionOutput, self).__init__(
                combined_embeddings = combined_embeddings,
                last_hidden_state = last_hidden_state,
                last_attention = last_attention,
                hidden_states = hidden_states,
                attentions = attentions,
                tasks_prediction = tasks_prediction,
                tasks_weighting = tasks_weighting)

        self._exception_keys = ('tasks_prediction', 'tasks_weighting', )


