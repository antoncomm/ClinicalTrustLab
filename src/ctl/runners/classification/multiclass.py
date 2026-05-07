import torch

from ctl.runners.base import Base


class MultiClassificationRunner(Base):

    def run_batch(self) -> None:
        self.optimizer.zero_grad()
        output = {"cls_logits": self.model(self.batch[0])}

        y = self.batch[1]["labels"][..., 0].to(dtype=torch.long).contiguous()
        self.batch[1]["labels"] = y

        loss = self.criterion(output["cls_logits"], self.batch[1]["labels"])
        self.total_loss += loss.sum().item()
        self.metrics.add_batch(predictions=output, references=self.batch[1])
        if self.is_train_dataset:
            self.engine.backward(loss)
            self.optimizer.step()
