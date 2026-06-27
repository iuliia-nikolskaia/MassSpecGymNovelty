import typing as T
from abc import ABC

import pandas as pd
import torch
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_recall
from torch_geometric.utils import unbatch

from massspecgym.models.base import MassSpecGymModel, Stage
import massspecgym.utils as utils

class NoveltyDetectionMassSpecGymModel(MassSpecGymModel, ABC):
    def __init__(
        self,
        at_ks: T.Iterable[int] = (500, 1000, 2000), 
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.at_ks = at_ks

        # Buffers to accumulate per-molecule scores/labels across all batches of an
        # epoch, since the novelty ranking is GLOBAL (one ranked list over the whole
        # val/test set) and can only be evaluated once the full epoch is collected.
        self._novelty_buffers: dict[Stage, dict[str, list]] = {
            Stage.VAL: {"scores": [], "labels": [], "identifier": []},
            Stage.TEST: {"scores": [], "labels": [], "identifier": []},
        }

# evaluates and logs novelty in the end of epoch, not per batch
    def evaluate_novelty(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        stage: Stage,
    ) -> dict[str, torch.Tensor]:
        
        metric_vals = {}
        # Evaluate precision and recall of the predicted novelty ranking against the
        # precomputed ground-truth novelty labels, at different top-k values
        for at_k in self.at_ks:
            precision_k = retrieval_precision(scores, labels, top_k=at_k)
            recall_k = retrieval_recall(scores, labels, top_k=at_k)

            for metric_name, val in [
                (f"{stage.to_pref()}precision@{at_k}", precision_k),
                (f"{stage.to_pref()}recall@{at_k}", recall_k),
            ]:
                self.log(metric_name, val)
                metric_vals[metric_name] = val

        return metric_vals

    def on_batch_end(
        self, outputs: T.Any, batch: dict, batch_idx: int, stage: Stage
    ) -> None:
        """
        Called at the end of every batch. Logs the loss (if any) and, for the val/test
        stages, accumulates the per-molecule novelty scores and ground-truth labels into
        per-stage buffers. The novelty ranking is global, so the actual metrics are not
        computed here but once at epoch end on the concatenated buffers.
        """

        loss = outputs.get("loss") if isinstance(outputs, dict) else None
        if loss is not None:
            self.log(
                f"{stage.to_pref()}loss",
                loss,
                batch_size=batch['spec'].size(0),
                sync_dist=True,
                prog_bar=True,
            )

        if stage in self.log_only_loss_at_stages or stage not in self._novelty_buffers:
            return

        # Accumulate this batch's scores/labels for the global epoch-end ranking
        buffer = self._novelty_buffers[stage]
        buffer["scores"].append(outputs["scores"].detach())
        buffer["labels"].append(batch["labels"].detach())
        if "identifier" in batch:
            buffer["identifier"].extend(batch["identifier"])

    def _on_epoch_end(self, stage):
        buffer = self._novelty_buffers[stage]
        scores = torch.cat(buffer["scores"]) 
        labels = torch.cat(buffer["labels"])

        self.evaluate_novelty(scores, labels, stage)

    def on_test_epoch_end(self):
        self._on_epoch_end(Stage.TEST)

    def on_validation_epoch_end(self):
        self._on_epoch_end(Stage.VAL)

    def _on_epoch_start(self, stage):
        self._novelty_buffers[stage] = {"scores": [], "labels": [], "identifier": []}

    def on_test_epoch_start(self):
        self._on_epoch_start(Stage.TEST)

    def on_validation_epoch_start(self):
            self._on_epoch_start(Stage.VAL)

    def get_checkpoint_monitors(self) -> list[dict]:
        monitors = [
            {"monitor": f"{Stage.VAL.to_pref()}precision@1000", "mode": "max", "early_stopping": True}
        ]
        return monitors