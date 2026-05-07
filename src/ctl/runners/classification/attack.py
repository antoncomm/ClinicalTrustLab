from .simple import SimpleClassificationRunner
from .multiclass import MultiClassificationRunner
from ctl.utils import attacks
from tqdm import tqdm
import torch


class AttackClassificationRunner(SimpleClassificationRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_batch(self) -> None:
        self.optimizer.zero_grad()
        x_adv, gt_labels = self.attack.generate(self.batch[0], self.batch[1]["labels"])
        self.output = {"cls_logits": self.model(x_adv)}
        loss = self.criterion(self.output["cls_logits"], gt_labels)
        self.total_loss += loss.sum().item()
        self.metrics.add_batch(
            predictions=self.output,
            references=self.batch[1],
            keep_in_ram=not self.is_train_dataset,
        )
        if self.is_train_dataset:
            # self.optimizer.zero_grad()
            self.engine.backward(loss)
            self.optimizer.step()

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        attack_params = {}
        if "attack_params" in self.config["attack"]:
            attack_params = self.config["attack"]["attack_params"]
        self.attack = getattr(attacks, self.config["attack"]["type"])(
            model=self.model, 
            mean=self.config["transforms"]["augmentations"]["Normalize"]["mean"], 
            std=self.config["transforms"]["augmentations"]["Normalize"]["std"],
            height=self.config["transforms"]["augmentations"]["Resize"]["height"],
            width=self.config["transforms"]["augmentations"]["Resize"]["width"],
            **attack_params,
        )

    def run_dataset(self) -> None:
        self.model.train(self.is_train_dataset)
        for self.batch in tqdm(
            self.dataset,
            disable=(not self.engine.is_local_main_process or not self.verbose),
        ):
            self._run_event("on_batch_start")
            self.run_batch()
            self._run_event("on_batch_end")


class AttackMultiClassificationRunner(MultiClassificationRunner):

    def run_batch(self) -> None:
        self.optimizer.zero_grad()
        y = self.batch[1]["labels"][..., 0].to(dtype=torch.long).contiguous()
        self.batch[1]["labels"] = y

        x_adv, gt_labels = self.attack.generate(self.batch[0], self.batch[1]["labels"])
        output = {"cls_logits": self.model(x_adv)}

        loss = self.criterion(output["cls_logits"], gt_labels)
        self.total_loss += loss.sum().item()
        self.metrics.add_batch(predictions=output, references=self.batch[1])
        if self.is_train_dataset:
            self.engine.backward(loss)
            self.optimizer.step()

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        attack_params = {}
        if "attack_params" in self.config["attack"]:
            attack_params = self.config["attack"]["attack_params"]
        self.attack = getattr(attacks, self.config["attack"]["type"])(
            model=self.model, 
            mean=self.config["transforms"]["augmentations"]["Normalize"]["mean"], 
            std=self.config["transforms"]["augmentations"]["Normalize"]["std"],
            height=self.config["transforms"]["augmentations"]["Resize"]["height"],
            width=self.config["transforms"]["augmentations"]["Resize"]["width"],
            **attack_params,
        )

    def run_dataset(self) -> None:
        self.model.train(self.is_train_dataset)
        for self.batch in tqdm(
            self.dataset,
            disable=(not self.engine.is_local_main_process or not self.verbose),
        ):
            self._run_event("on_batch_start")
            self.run_batch()
            self._run_event("on_batch_end")
