from .simple import SimpleClassificationRunner
from neurone.utils import attacks


class AttackClassificationRunner(SimpleClassificationRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.attack = getattr(attacks, self.config["run"]["attack_type"])(model=self.model)

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
