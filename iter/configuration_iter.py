from __future__ import annotations

from typing import Dict, Any, Optional

import transformers
from transformers import PretrainedConfig, AutoConfig, T5Tokenizer, BertTokenizerFast, T5TokenizerFast, BertTokenizer, \
    BertModel, LongT5EncoderModel, T5EncoderModel, AlbertTokenizerFast, AlbertTokenizer, AlbertModel, \
    RobertaTokenizerFast, RobertaTokenizer, RobertaModel, AutoTokenizer

from iter.modeling_features import FeaturesMixin


class ITERConfig(PretrainedConfig, FeaturesMixin):
    d_ff: int
    d_model: int
    num_types: int
    num_links: int
    features: int
    max_nest_depth: int
    dropout: float
    threshold: float
    transformer_config: PretrainedConfig
    model_type = "iter"

    def __init__(
            self,
            transformer_name="t5-small",
            transformer_config=None,
            num_types=4,
            num_links=5,
            features=0,
            dataset: Optional[str] = None,
            max_nest_depth: int = 1,
            dropout: float = 0.3,
            activation_fn: str = "gelu",
            use_gate: bool = True,
            use_bias: bool = False,
            use_mlp: bool = True,
            d_ff: int = 0,
            threshold: float = 0.5,
            entity_types: Optional[list[str]] = None,
            link_types: Optional[list[str]] = None,
            # --- New parameters for Wenyanwen Geo-NER enhancements ---
            use_fact_decomposition: bool = False,
            use_multiscale_geo: bool = False,
            use_gradient_isolation: bool = True,
            num_iterations: int = 3,
            fact_loss_weight: float = 0.5,
            multiscale_loss_weight: float = 0.1,
            num_bilinear_groups: int = 4,
            cnn_kernel_sizes: Optional[list[int]] = None,
            max_span_length: int = 30,
            long_context_chunk_size: int = 32,
            long_context_num_heads: int = 8,
            long_context_dropout: float = 0.1,
            ner_type_loss_weight: float = 0.0,
            ner_type_class_weights: Optional[list[float]] = None,
            ner_type_label_smoothing: float = 0.0,
            ner_type_focal_gamma: float = 0.0,
            ner_type_focus_types: Optional[list[str]] = None,
            ner_type_focus_weight: float = 1.0,
            ner_type_decode_weight: float = 0.0,
            ner_boundary_decode_weight: float = 0.0,
            ner_boundary_start_candidate_fusion: bool = False,
            ner_boundary_loss_weight: float = 0.0,
            ner_boundary_consistency_loss_weight: float = 0.0,
            ner_boundary_hard_negative_weight: float = 2.0,
            ner_boundary_adjacent_negative_weight: Optional[float] = None,
            ner_boundary_internal_negative_weight: Optional[float] = None,
            ner_boundary_positive_weight: float = 1.0,
            ner_boundary_hard_negative_radius: int = 1,
            ner_boundary_focal_gamma: float = 0.0,
            # Low-rank biaffine span scorer.  It is an additive residual to
            # the legacy ITER span head, so old checkpoints remain loadable.
            ner_span_biaffine: bool = False,
            ner_span_biaffine_rank: int = 64,
            ner_span_biaffine_weight: float = 0.35,
            ner_span_width_embedding: bool = True,
            ner_span_learned_gate: bool = False,
            ner_span_margin_loss_weight: float = 0.0,
            ner_span_margin: float = 0.20,
            ner_span_margin_max_negatives: int = 8,
            ner_span_margin_hard_negative_radius: int = 2,
            ner_type_prototype_loss_weight: float = 0.0,
            ner_type_prototype_decode_weight: float = 0.0,
            ner_type_prototype_temperature: float = 0.15,
            ner_type_confusion_margin: float = 0.20,
            ner_type_confusion_pairs: Optional[list[list[str]]] = None,
            ner_target_adapter: bool = False,
            ner_target_adapter_size: int = 128,
            ner_target_adapter_dropout: float = 0.10,
            ner_target_adapter_scale: float = 0.30,
            **kwargs
    ):
        if isinstance(features, list):
            features = sum(2** feat for feat in features)
        self.features = features

        transformer_name = (transformer_config or {}).get("_name_or_path", transformer_name)
        # DEBUG: print what we're trying to load
        import os as _os
        _offline = _os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
        print(f"  [DEBUG ITERConfig.__init__] transformer_name={transformer_name!r}, "
              f"has_transformer_config={'model_type' in (transformer_config or {})}, "
              f"offline={_offline}")

        # When the full transformer config dict is available (always the case
        # when loading from a saved ITER checkpoint), reconstruct it locally
        # to avoid requiring HF access.  This is essential for offline mode.
        if transformer_config and 'model_type' in transformer_config:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            try:
                config_cls = CONFIG_MAPPING[transformer_config['model_type']]
            except KeyError:
                config_cls = PretrainedConfig
            self.transformer_config = config_cls(**transformer_config)
            print(f"  [DEBUG ITERConfig.__init__] reconstructed {config_cls.__name__} locally")
        elif transformer_config:
            # Partial config dict — try HF, fall back to a minimal placeholder
            try:
                self.transformer_config = AutoConfig.from_pretrained(
                    transformer_name, local_files_only=_offline, **transformer_config
                )
            except (OSError, EnvironmentError, ValueError) as e:
                print(f"  [DEBUG ITERConfig.__init__] HF load failed ({e}), using PretrainedConfig placeholder")
                self.transformer_config = PretrainedConfig(**transformer_config)
        else:
            # No config dict (e.g. bare ITERConfig() for to_diff_dict).
            # Try HF, but if offline/unavailable use a minimal placeholder.
            try:
                self.transformer_config = AutoConfig.from_pretrained(
                    transformer_name, local_files_only=_offline
                )
            except (OSError, EnvironmentError, ValueError) as e:
                print(f"  [DEBUG ITERConfig.__init__] bare init: HF load failed ({e}), using PretrainedConfig placeholder")
                # Infer model_type from the name to pick the right config class
                _model_type = None
                for _candidate in ('t5', 'bert', 'albert', 'roberta', 'longt5'):
                    if _candidate in transformer_name.lower():
                        _model_type = _candidate
                        break
                if _model_type:
                    try:
                        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                        self.transformer_config = CONFIG_MAPPING[_model_type]()
                    except (KeyError, Exception):
                        self.transformer_config = PretrainedConfig(model_type=_model_type or 'bert')
                else:
                    self.transformer_config = PretrainedConfig()
        self.num_types = num_types
        self.num_links = num_links
        self.dataset = dataset
        self.max_nest_depth = max_nest_depth
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.use_bias = use_bias
        self.use_gate = use_gate
        self.use_mlp = use_mlp
        self.use_scale = d_ff == 0
        self.threshold = threshold
        self.entity_types = entity_types or list()
        self.link_types = link_types or list()
        # --- Store new Wenyanwen Geo-NER parameters ---
        self.use_fact_decomposition = use_fact_decomposition
        self.use_multiscale_geo = use_multiscale_geo
        self.use_gradient_isolation = use_gradient_isolation
        self.num_iterations = num_iterations
        self.fact_loss_weight = fact_loss_weight
        self.multiscale_loss_weight = multiscale_loss_weight
        self.num_bilinear_groups = num_bilinear_groups
        self.cnn_kernel_sizes = cnn_kernel_sizes or [2, 3, 4, 5]
        self.max_span_length = max_span_length
        self.long_context_chunk_size = long_context_chunk_size
        self.long_context_num_heads = long_context_num_heads
        self.long_context_dropout = long_context_dropout
        self.ner_type_loss_weight = ner_type_loss_weight
        self.ner_type_class_weights = ner_type_class_weights
        self.ner_type_label_smoothing = ner_type_label_smoothing
        self.ner_type_focal_gamma = ner_type_focal_gamma
        self.ner_type_focus_types = ner_type_focus_types or []
        self.ner_type_focus_weight = ner_type_focus_weight
        self.ner_type_decode_weight = ner_type_decode_weight
        self.ner_boundary_decode_weight = ner_boundary_decode_weight
        self.ner_boundary_start_candidate_fusion = ner_boundary_start_candidate_fusion
        self.ner_boundary_loss_weight = ner_boundary_loss_weight
        self.ner_boundary_consistency_loss_weight = ner_boundary_consistency_loss_weight
        self.ner_boundary_hard_negative_weight = ner_boundary_hard_negative_weight
        self.ner_boundary_adjacent_negative_weight = ner_boundary_adjacent_negative_weight
        self.ner_boundary_internal_negative_weight = ner_boundary_internal_negative_weight
        self.ner_boundary_positive_weight = ner_boundary_positive_weight
        self.ner_boundary_hard_negative_radius = ner_boundary_hard_negative_radius
        self.ner_boundary_focal_gamma = ner_boundary_focal_gamma
        self.ner_span_biaffine = ner_span_biaffine
        self.ner_span_biaffine_rank = ner_span_biaffine_rank
        self.ner_span_biaffine_weight = ner_span_biaffine_weight
        self.ner_span_width_embedding = ner_span_width_embedding
        self.ner_span_learned_gate = ner_span_learned_gate
        self.ner_span_margin_loss_weight = ner_span_margin_loss_weight
        self.ner_span_margin = ner_span_margin
        self.ner_span_margin_max_negatives = ner_span_margin_max_negatives
        self.ner_span_margin_hard_negative_radius = ner_span_margin_hard_negative_radius
        self.ner_type_prototype_loss_weight = ner_type_prototype_loss_weight
        self.ner_type_prototype_decode_weight = ner_type_prototype_decode_weight
        self.ner_type_prototype_temperature = ner_type_prototype_temperature
        self.ner_type_confusion_margin = ner_type_confusion_margin
        self.ner_type_confusion_pairs = ner_type_confusion_pairs or []
        self.ner_target_adapter = ner_target_adapter
        self.ner_target_adapter_size = ner_target_adapter_size
        self.ner_target_adapter_dropout = ner_target_adapter_dropout
        self.ner_target_adapter_scale = ner_target_adapter_scale
        # --- End new parameters ---
        self.d_ff = d_ff or self.try_attr_options(
            "d_ff",  # T5 model family
            "intermediate_size",  # BERT model family
            "encoder_ffn_dim",  # BART model family
        )
        self.d_model = self.try_attr_options(
            "d_model",  # T5
            "hidden_size",  # BERT
        )

        # Saved Hugging Face configs already contain this field. Supplying it
        # explicitly as well as through kwargs breaks checkpoint reloads.
        kwargs.setdefault("is_encoder_decoder", False)
        PretrainedConfig.__init__(self, **kwargs)
        self.max_length = self.transformer_config.max_length

    @property
    def num_real_links(self):
        return len(self.link_types)

    @property
    def num_real_types(self):
        return len(self.entity_types)

    @property
    def model_kwargs(self) -> dict:
        if self.is_feature_use_t5_decoder and self.transformer_config.model_type not in {"bert", "longt5"}:
            return dict(is_decoder=self.is_feature_use_t5_decoder_as_decoder)
        return dict()

    def tokenizer_kwargs(self, use_fast=True) -> dict:
        tokenizer_cls = self.guess_tokenizer_class(use_fast)
        tokenizer_kwargs = dict()
        if tokenizer_cls == AutoTokenizer:
            tokenizer_kwargs['use_fast'] = use_fast
        if "roberta" in self.transformer_config.model_type or "bart" in self.transformer_config.model_type:
            tokenizer_kwargs['add_prefix_space'] = True
        return tokenizer_kwargs

    def guess_tokenizer_class(self, use_fast=False):
        if "t5" in self.transformer_config.model_type:
            return T5Tokenizer if not use_fast else T5TokenizerFast
        elif "albert" in self.transformer_config.model_type:
            return AlbertTokenizer if not use_fast else AlbertTokenizerFast
        elif "roberta" in self.transformer_config.model_type:
            return RobertaTokenizer if not use_fast else RobertaTokenizerFast
        return AutoTokenizer

    def guess_model_class(self):
        if "bert" == self.transformer_config.model_type:
            return BertModel
        elif "longt5" == self.transformer_config.model_type:
            return LongT5EncoderModel
        elif "albert" == self.transformer_config.model_type:
            return AlbertModel
        elif "roberta" in self.transformer_config.model_type:
            return RobertaModel
        elif "t5" in self.transformer_config.model_type:
            if self.is_feature_use_t5_decoder:
                raise NotImplementedError
            else:
                return T5EncoderModel
        else:
            return transformers.AutoModel

    def try_attr_options(self, *items):
        exceptions = []
        for item in items:
            try:
                return self.transformer_config.__getattribute__(item)
            except AttributeError as cause:
                exceptions.append(cause)
        raise AttributeError from exceptions[-1]

    def to_diff_dict(self) -> Dict[str, Any]:
        """Override to prevent ``to_diff_dict`` from creating a bare
        ``ITERConfig()`` instance, which would trigger an expensive /
        impossible HF download in offline mode.

        Instead of comparing against a default instance, we simply
        return the full config dict (equivalent to setting
        ``is_composition = True``).
        """
        config_dict = self.to_dict()
        serializable_config_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, PretrainedConfig):
                serializable_config_dict[key] = value.to_diff_dict()
            else:
                serializable_config_dict[key] = value
        return serializable_config_dict

    @staticmethod
    def _get_generation_defaults() -> Dict[str, Any]:
        return {}

    def _get_non_default_generation_parameters(self) -> Dict[str, Any]:
        # hacky workaround as transformers errors out if you have overridden `max_length` as of v44
        return dict()
