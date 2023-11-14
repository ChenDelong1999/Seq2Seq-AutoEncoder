# Modified from https://huggingface.co/docs/transformers/main/model_doc/time_series_transformer
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_seq2seq_autoencoder": [
        "SEQ2SEQ_AUTOENCODER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Seq2SeqAutoEncoderConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_seq2seq_autoencoder"] = [
        "SEQ2SEQ_AUTOENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Seq2SeqAutoEncoderModel",
        "Seq2SeqAutoEncoderPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_seq2seq_autoencoder import (
        SEQ2SEQ_AUTOENCODER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Seq2SeqAutoEncoderConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_seq2seq_autoencoder import (
            SEQ2SEQ_AUTOENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TimeSeriesTransformerForPrediction,
            Seq2SeqAutoEncoderModel,
            Seq2seqAutoencoderPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
