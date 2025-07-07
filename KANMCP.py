from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
import global_configs
from modules.LinearEncoder import LinearEncoder
from modules.mib import *
from kan import *
from global_configs import DEVICE
from modules.transformer import TransformerEncoder


class KANMCP(DebertaV2PreTrainedModel):
    def __init__(self, deberta_config, multimodal_config):
        super().__init__(deberta_config)
        self.TEXT_DIM = global_configs.TEXT_DIM
        self.ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
        self.VISUAL_DIM = global_configs.VISUAL_DIM
        
        self.seed = multimodal_config.seed
        self.compressed_dim = multimodal_config.compressed_dim
        self.kan_hidden_neurons = multimodal_config.kan_hidden_neurons

        # Deberta
        Deberta = DebertaV2Model.from_pretrained(multimodal_config.model)
        self.Deberta = Deberta.to(DEVICE)

        # DRD-MIB
        self.TEncoder = mib(self.TEXT_DIM, self.compressed_dim, multimodal_config.m_dim)
        self.AEncoder = mib(self.ACOUSTIC_DIM, self.compressed_dim, multimodal_config.m_dim)
        self.VEncoder = mib(self.VISUAL_DIM, self.compressed_dim, multimodal_config.m_dim)

        self.KAN = KAN(width=[self.compressed_dim * 3, self.kan_hidden_neurons, 1], device="cuda", auto_save=False, seed=self.seed)

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            label_ids
    ):
        # deberta processing text data
        embedding_output = self.Deberta(input_ids)
        x_embedding = embedding_output[0]

        # feature extraction
        x = torch.mean(x_embedding, dim=1)
        a = torch.mean(acoustic, dim=1)
        v = torch.mean(visual, dim=1)

        # feature reduction
        text_feature, out_t, loss_t = self.TEncoder(x, label_ids)
        audio_feature, out_a, loss_a = self.AEncoder(a, label_ids)
        visual_feature, out_v, loss_v = self.VEncoder(v, label_ids)

        # concat
        concat_feature = torch.cat([text_feature, audio_feature, visual_feature], dim=1)

        # fusion and predict
        logits = self.KAN(concat_feature)

        res = {
            "logits": logits,
            "text_feature": text_feature,
            "audio_feature": audio_feature,
            "visual_feature": visual_feature,
            "concat_feature": concat_feature,
            "loss_t": loss_t,
            "loss_a": loss_a,
            "loss_v": loss_v,
            "out_t": out_t,
            "out_a": out_a,
            "out_v": out_v
        }

        return res
