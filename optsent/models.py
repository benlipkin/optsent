import functools

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from optsent.abstract import Object


class Model(Object):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._id = model_id
        try:
            self._config = AutoConfig.from_pretrained(self._id)
            self._tokenizer = AutoTokenizer.from_pretrained(self._id)
            self._model = AutoModelForCausalLM.from_pretrained(self._id)
            self._model.eval()
        except Exception as invalid_id:
            raise ValueError(
                "model must be valid HuggingFace CausalLM."
            ) from invalid_id
        self._set_torch_device()
        self.log(f"Loaded pretrained {self._id} model on {self._device}.")

    def _set_torch_device(self) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
            try:
                self._model = self._model.to(self._device)
                return
            except RuntimeError:
                self._device = torch.device("cpu")
                torch.set_default_tensor_type(torch.FloatTensor)
                self._model = self._model.to(self._device)
        else:
            self._device = torch.device("cpu")
            self._model = self._model.to(self._device)

    @functools.cache
    def score(self, sent: str) -> float:
        if not isinstance(sent, str):
            raise TypeError("sent must be type `str` to get scored.")
        with torch.no_grad():
            inputs = self._tokenizer(sent, return_tensors="pt").to(self._device)
            tokens = inputs["input_ids"]
            outputs = self._model(**inputs, labels=tokens)
            loss = torch.nn.CrossEntropyLoss(reduction="none")(
                outputs.logits[..., :-1, :]
                .contiguous()
                .view(-1, outputs.logits.size(-1)),
                tokens[..., 1:].contiguous().view(-1),
            ).view(tokens.size(0), tokens.size(-1) - 1)
            loss = (loss * inputs["attention_mask"][..., 1:].contiguous()).sum(dim=1)
            logp = -loss.cpu().detach().item()
        return logp

    @functools.cache
    def embed(self, sent: str) -> npt.NDArray[np.float32]:
        if not isinstance(sent, str):
            raise TypeError("sent must be type `str` to get embed.")
        with torch.no_grad():
            inputs = self._tokenizer(sent, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(axis=1).cpu().detach().numpy()
        return embedding
