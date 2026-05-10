from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from torch import nn


class MultiTaskConfig(PretrainedConfig):
    model_type = "multitask_classifier"

    def __init__(
        self,
        model_name="l3cube-pune/hing-roberta",
        dropout_prob=0.15,
        num_aggression_labels=3,
        num_offense_labels=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.dropout_prob = dropout_prob
        self.num_aggression_labels = num_aggression_labels
        self.num_offense_labels = num_offense_labels


class MultiTaskClassifier(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(
        self,
        config=None,
        model_name="l3cube-pune/hing-roberta",
        dropout_prob=0.15,
    ):
        if config is None:
            config = MultiTaskConfig(model_name=model_name, dropout_prob=dropout_prob)
        super().__init__(config)

        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout_prob)
        self.aggression_head = nn.Linear(hidden_size, config.num_aggression_labels)
        self.offense_head = nn.Linear(hidden_size, config.num_offense_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs, **kwargs)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {
            "aggression": self.aggression_head(pooled),
            "offense": self.offense_head(pooled),
        }
