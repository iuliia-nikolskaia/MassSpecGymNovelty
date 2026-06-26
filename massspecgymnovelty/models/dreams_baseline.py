import torch
import torch.nn.functional as F

from massspecgym.models.base import Stage
from massspecgymnovelty.models.base import NoveltyDetectionMassSpecGymModel


class DreaMSNoveltyBaseline(NoveltyDetectionMassSpecGymModel):

    def __init__(self, embedding_key="DreaMS_embedding", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_key = embedding_key
        self.automatic_optimization = False
        self.register_buffer("train_embeddings", torch.empty(0), persistent=True)
        self._train_chunks = []

    def configure_optimizers(self):
        return None

    def _get_embeddings(self, batch):
        emb = batch.get(self.embedding_key, batch.get("spec"))
        return F.normalize(emb.float(), dim=-1)

    def _freeze_train_embeddings(self):
        if self._train_chunks:
            self.train_embeddings = torch.cat(self._train_chunks, dim=0)
            self._train_chunks = []

    def on_train_epoch_start(self):
        self._train_chunks = []

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._freeze_train_embeddings()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._freeze_train_embeddings()

    def step(self, batch, stage=Stage.NONE):
        emb = self._get_embeddings(batch)

        # train: just collect the embeddings to freeze, no loss/scores
        if stage == Stage.TRAIN:
            self._train_chunks.append(emb.detach())
            return {}

        if self.train_embeddings.numel() == 0:
            return {"scores": torch.zeros(emb.size(0), device=emb.device)}

        sims = emb @ self.train_embeddings.to(emb.device).T
        return {"scores": 1.0 - sims.max(dim=1).values}
