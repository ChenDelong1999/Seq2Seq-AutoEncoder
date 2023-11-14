# Modified from https://huggingface.co/docs/transformers/main/model_doc/time_series_transformer
from typing import List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

SEQ2SEQ_AUTOENCODER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/time-series-transformer-tourism-monthly": (
        "https://huggingface.co/huggingface/time-series-transformer-tourism-monthly/resolve/main/config.json"
    ),
    # See all TimeSeriesTransformer models at https://huggingface.co/models?filter=time_series_transformer
}


class Seq2SeqAutoEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TimeSeriesTransformerModel`]. It is used to
    instantiate a Time Series Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Time Series
    Transformer
    [huggingface/time-series-transformer-tourism-monthly](https://huggingface.co/huggingface/time-series-transformer-tourism-monthly)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        prediction_length (`int`):
            The prediction length for the decoder. In other words, the prediction horizon of the model. This value is
            typically dictated by the dataset and we recommend to set it appropriately.
        model_seq_length (`int`, *optional*, defaults to `prediction_length`):
            The context length for the encoder. If `None`, the context length will be the same as the
            `prediction_length`.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model. Could be either "student_t", "normal" or "negative_binomial".
        loss (`string`, *optional*, defaults to `"nll"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood (nll) - which currently is the only supported one.
        input_channels (`int`, *optional*, defaults to 1):
            The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
            multivariate targets.
        scaling (`string` or `bool`, *optional* defaults to `"mean"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        lags_sequence (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 5, 6, 7]`):
            The lags of the input time series as covariates often dictated by the frequency of the data. Default is
            `[1, 2, 3, 4, 5, 6, 7]` but we recommend to change it based on the dataset appropriately.
        num_time_features (`int`, *optional*, defaults to 0):
            The number of time features in the input time series.
        num_dynamic_real_features (`int`, *optional*, defaults to 0):
            The number of dynamic real valued features.
        num_static_categorical_features (`int`, *optional*, defaults to 0):
            The number of static categorical features.
        num_static_real_features (`int`, *optional*, defaults to 0):
            The number of static real valued features.
        cardinality (`list[int]`, *optional*):
            The cardinality (number of different values) for each of the static categorical features. Should be a list
            of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        embedding_dimension (`list[int]`, *optional*):
            The dimension of the embedding for each of the static categorical features. Should be a list of integers,
            having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        d_model (`int`, *optional*, defaults to 64):
            Dimensionality of the transformer layers.
        encoder_layers (`int`, *optional*, defaults to 2):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 2):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and
            `"relu"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder, and decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each encoder layer.
        decoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each decoder layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability used between the two layers of the feed-forward networks.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for each time step of inference.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.

        Example:

    ```python
    >>> from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

    >>> # Initializing a Time Series Transformer configuration with 12 time steps for prediction
    >>> configuration = TimeSeriesTransformerConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = TimeSeriesTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "sequence_to_sequence_autoencoder"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    def __init__(
        self,
        data_seq_length: int = 1024,
        loss: str = "mse",
        input_channels: int = 1,
        encoder_ffn_dim: int = 32,
        decoder_ffn_dim: int = 32,
        encoder_attention_heads: int = 2,
        decoder_attention_heads: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation_function: str = "gelu",
        d_model: int = 64,
        dropout: float = 0.1,
        encoder_layerdrop: float = 0.1,
        decoder_layerdrop: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        init_std: float = 0.02,
        num_queries: int = 64,
        d_latent: int = 1024,
        **kwargs,
    ):
        # assert data_length is not None and int(data_length ** 0.5) ** 2 == data_length, "data_length must be a square number"

        # time series specific configuration
        self.data_seq_length = data_seq_length
        self.model_seq_length = data_seq_length + num_queries + 1
        self.loss = loss
        self.input_channels = input_channels
        self.num_queries = num_queries
        self.d_latent = d_latent

        # Transformer architecture configuration
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop

        self.activation_function = activation_function
        self.init_std = init_std

        self.use_cache = True

        super().__init__(is_encoder_decoder=True, **kwargs)

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_time_features
            + self.num_static_real_features
        )
