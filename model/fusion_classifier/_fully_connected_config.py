from model import MyModelConfig


class FCFullyConnectedModelConfig(MyModelConfig):
    def __init__(
            self,
            model_dir: str,
            model_name: str = 'model',
            n_classes: int = 1,
            hidden_size: int = 1024,
            n_hidden_layers: int = 1,
            pooling_op: str = 'flatten',
            **kwargs
    ):
        # check no. classes
        assert n_classes > 1
        # check pooling operation
        assert pooling_op in ['flatten', 'max', 'mean', 'add']
        # call super class
        super().__init__(
            model_dir=model_dir,
            model_name=model_name,
            hyper_parameters={
                'hidden_size': hidden_size,
                'n_hidden_layers': n_hidden_layers,
                'pooling_op': pooling_op
            },
            **kwargs
        )

    @property
    def hidden_size(self) -> int:
        return self.hyper_parameters['hidden_size']

    @property
    def n_hidden_layers(self) -> int:
        return self.hyper_parameters['n_hidden_layers']

    @property
    def pooling_op(self) -> str:
        return self.hyper_parameters['pooling_op']
