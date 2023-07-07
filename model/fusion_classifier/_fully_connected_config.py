from model import MyModelConfig


class FCFullyConnectedModelConfig(MyModelConfig):
    def __init__(
            self,
            gene_classifier_name: str,
            gene_classifier_path: str,
            freeze: bool = True,
            hidden_size: int = 1024,
            n_hidden_layers: int = 1,
            pooling_op: str = 'flatten',
            **kwargs
    ):
        # check pooling operation
        assert pooling_op in ['flatten', 'max', 'mean', 'add']
        # call super class
        super().__init__(
            hyper_parameters={
                'gene_classifier_name': gene_classifier_name,
                'gene_classifier_path': gene_classifier_path,
                'freeze': freeze,
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
