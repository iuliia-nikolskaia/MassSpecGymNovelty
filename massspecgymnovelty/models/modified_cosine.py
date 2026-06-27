import torch

from matchms.similarity import ModifiedCosine
from matchms import calculate_scores
from massspecgym.models.base import Stage
from massspecgymnovelty.models.base import NoveltyDetectionMassSpecGymModel


class ModifiedCosineBaseline(NoveltyDetectionMassSpecGymModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.train_spectra = []
        self.id_to_spec = {}

    def configure_optimizers(self):
        return None

    def _build_id_to_spec(self):
        if not self.id_to_spec:
            ds = self.trainer.datamodule.dataset
            self.id_to_spec = dict(zip(ds.metadata["identifier"].astype(str), ds.spectra))

    def _get_matchms_spectra(self, batch):
        self._build_id_to_spec()
        return [self.id_to_spec[str(i)] for i in batch["identifier"]]

    def _prepare_train_spectra(self):
        if self.train_spectra:
            return
        self._build_id_to_spec()
        dm = self.trainer.datamodule
        train_ids = dm.dataset.metadata.iloc[dm.train_dataset.indices]["identifier"].astype(str)
        self.train_spectra = [self.id_to_spec[i] for i in train_ids]

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._prepare_train_spectra()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._prepare_train_spectra()

    def step(self, batch, stage=Stage.NONE):
        if stage == Stage.TRAIN:
            return {}

        spectra = self._get_matchms_spectra(batch)
        device = batch["labels"].device
        if not self.train_spectra:
            return {"scores": torch.zeros(len(spectra), device=device)}

        sims = calculate_scores(spectra, self.train_spectra, ModifiedCosine())
        max_sim = sims.to_array()["ModifiedCosine_score"].max(axis=1)
        return {"scores": torch.as_tensor(1.0 - max_sim, dtype=torch.float32, device=device)}
