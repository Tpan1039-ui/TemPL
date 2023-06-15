from transformers.models.esm.modeling_esm import EsmPreTrainedModel, EsmModel, EsmLMHead, EsmConfig
import torch.nn as nn
import torch


def make_config():
    config = EsmConfig(
        vocab_size=33,
        mask_token_id=32,
        pad_token_id=1,
        hidden_size=1280,
        num_hidden_layers=33,
        num_attention_heads=20,
        intermediate_size=5120,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        position_embedding_type="rotary",
        emb_layer_norm_before=False,
        token_dropout=True,
    )
    return config


class AttentionHead(nn.Module):
    def __init__(self, config):
        super(AttentionHead, self).__init__()
        self.attn = nn.Linear(config.hidden_size, 1)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, 1),
        )

    def forward(self, features, **kwargs):
        x = self.norm(features)
        context_vector = torch.einsum("bse,bs->be", x, torch.softmax(self.attn(x)[:, :, 0], dim=-1))
        return self.mlp(context_vector)


class TemPL(EsmPreTrainedModel):

    @classmethod
    def load(cls, model_name):
        if model_name == "templ-base":
            model = TemPL(ogt_head=True)
            checkpoint_path = "templ-base.ckpt"
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            model.eval()
            return model

        elif model_name == "templ-tm-fine-tuning":
            model = TemPL(ogt_head=False)
            checkpoint_path = "templ-tm-fine-tuning.ckpt"
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            model.eval()
            return model

        else:
            raise ValueError(f"Unknown model name {model_name}.")

    def __init__(self, config=None, ogt_head=False):
        if config is None:
            config = make_config()
        super(TemPL, self).__init__(config)
        self.esm = EsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        if ogt_head:
            self.classifier = AttentionHead(config)
        # self.init_weights()

    def forward(self, input_ids, attention_mask=None, task="ogt_prediction"):
        assert task in ["ogt_prediction",
                        "mask_prediction"], f"The task must be one of ['ogt_prediction', 'mask_prediction'], but got {task}."
        features = self.esm(input_ids, attention_mask=attention_mask)[0]
        if task == "ogt_prediction":
            return self.classifier(features)
        elif task == "mask_prediction":
            return self.lm_head(features)
        else:
            raise ValueError(f"Unknown task {task}.")