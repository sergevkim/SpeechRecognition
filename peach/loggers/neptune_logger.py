#https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/neptune.py
import neptune


class NeptuneLogger:
    def __init__(
            self,
            api_key,
            project_name,
        ):

    self.api_key = api_key
    self.project_name = project_name

