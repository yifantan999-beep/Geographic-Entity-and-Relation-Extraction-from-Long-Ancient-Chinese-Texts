from transformers import BertConfig

from iter.configuration_iter import ITERConfig


def test_saved_is_encoder_decoder_can_be_loaded_from_kwargs():
    transformer_config = BertConfig().to_dict()
    transformer_config["_name_or_path"] = "local-bert"
    transformer_config["max_length"] = 256

    config = ITERConfig(
        transformer_name="local-bert",
        transformer_config=transformer_config,
        is_encoder_decoder=False,
    )

    assert config.is_encoder_decoder is False


def test_gradient_isolation_is_retained_in_iter_config():
    transformer_config = BertConfig().to_dict()
    transformer_config["_name_or_path"] = "local-bert"

    config = ITERConfig(
        transformer_name="local-bert",
        transformer_config=transformer_config,
        use_gradient_isolation=False,
    )

    assert config.use_gradient_isolation is False
