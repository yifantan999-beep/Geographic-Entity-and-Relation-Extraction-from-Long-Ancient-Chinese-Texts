feature_descriptions = dict()

FEATURE_NER_ONLY_BIT = 11


class FeaturesMixin:
    def __init__(self):
        self.features = 0

    # Note: NOT a staticmethod — called as a decorator factory during class definition
    # @feature_info(bit, desc) on property methods
    def feature_info(bit: int, desc: str):
        def decorator(func):
            feature_descriptions[getattr(func, "__name__")] = (bit, desc)
            return func
        return decorator

    def is_bit_enabled(self, offset: int) -> bool:
        return (self.features & (1 << offset)) != 0

    @property
    @feature_info(0, "Use an hidden state produced by an earlier layer in the Transformer for NER")
    def is_feature_offset_ner_hidden_state(self):
        return self.features & (1 << 0) != 0

    @property
    @feature_info(1, "Sum representations")
    def is_feature_sum_representations(self):
        return self.features & (1 << 1) != 0

    @property
    @feature_info(2, "Allow examples with no relations")
    def is_feature_examples_without_links(self):
        return self.features & (1 << 2) == 0

    @property
    @feature_info(3,
                  "Extended context window. ACE05, when preprocessed as in ASP, supplies context around "
                  "sentences. This flag enables using said context in training")
    def is_feature_use_extended_context(self):
        return self.features & (1 << 3) != 0

    @property
    @feature_info(4, "Use LSE loss for rr_score")
    def is_feature_use_lse_rr_loss(self):
        return self.features & (1 << 4) != 0

    @property
    @feature_info(5, "Extra LR class")
    def is_feature_extra_lr_class(self):
        return self.features & (1 << 5) != 0

    @property
    @feature_info(6, "Use T5 decoder parameters as a decoder")
    def is_feature_use_t5_decoder_as_decoder(self):
        flag = self.features & (1 << 6) != 0
        assert not flag or self.is_feature_use_t5_decoder, f"feature {self.is_feature_use_t5_decoder=} must be enabled"
        return flag

    @property
    @feature_info(7, "Extra RR class")
    def is_feature_extra_rr_class(self):
        return self.features & (1 << 7) != 0

    @property
    @feature_info(8, "Average the loss by batch size")
    def is_feature_batch_average_loss(self):
        # unused in the code but used in the individual datasets
        return self.features & (1 << 8) != 0

    @property
    @feature_info(9, "Nest depth > 1")
    def is_feature_nest_depth_gt_1(self):
        return self.features & (1 << 9) != 0

    @property
    @feature_info(10, "Optimized inference")
    def is_feature_perf_optimized(self):
        return self.features & (1 << 10) != 0

    @property
    @feature_info(11, "NER Only")
    def is_feature_ner_only(self):
        return self.features & (1 << 11) != 0

    @property
    @feature_info(12, "Use T5 decoder parameters as additional encoder")
    def is_feature_use_t5_decoder(self):
        return self.features & (1 << 12) != 0

    @property
    @feature_info(13, "Empty examples")
    def is_feature_empty_examples(self):
        return self.features & (1 << 13) != 0

    @property
    @feature_info(14, "Use negative samples for LSE rr loss during training")
    def is_feature_negative_lse_examples(self):
        return self.features & (1 << 14) != 0

    @property
    @feature_info(15, "Use CE loss for RR pair training")
    def is_feature_ce_loss(self):
        return self.features & (1 << 15) != 0

    @property
    @feature_info(16, "Train symmetric relations to be predicted in both directions")
    def is_feature_train_symrels_both_directions(self):
        return self.features & (1 << 16) != 0

    @property
    @feature_info(17, "Mimic PL Marker evaluation")
    def is_feature_mimic_pl_marker_eval(self):
        assert self.features & (1 << 17) == 0 or self.is_feature_train_symrels_both_directions, (
            f"To mimic PL marker eval, bit 16 (train_symrels_both_directions) must be enabled, "
            f"otherwise the evaluation will not be similar."
        )
        return self.features & (1 << 17) != 0

    @property
    @feature_info(18, "Negative samples from inverted link types")
    def is_feature_negsample_link_types(self):
        return self.features & (1 << 18) != 0

    @property
    @feature_info(19, "Make the model behave like PL Marker and output symmetric relations twice, always")
    def is_feature_behave_like_plmarker(self):
        return self.features & (1 << 19) != 0

    @property
    @feature_info(20, "Development flag")
    def is_feature_in_development(self):
        return self.features & (1 << 20) != 0

    @property
    @feature_info(21, "Dont create the rr_score Module")
    def is_feature_dont_set_rr_score(self):
        return self.features & (1 << 21) != 0

    # --- New Feature Bits for Wenyanwen Geo-NER Enhancements ---

    @property
    @feature_info(22, "Enable Fact Decomposition module (Stage 1: span extraction + Stage 2: relation verification)")
    def is_feature_fact_decomposition(self):
        """When enabled, adds the fact decomposition module for two-stage entity/relation extraction."""
        return self.features & (1 << 22) != 0

    @property
    @feature_info(23, "Enable Multi-Scale Geographic Representation module")
    def is_feature_multiscale_geo(self):
        """When enabled, adds the multi-scale geographic representation module with
        character-level, n-gram, and span-level features via gated fusion."""
        return self.features & (1 << 23) != 0

    @property
    @feature_info(24, "Enable iterative fact-aware refinement")
    def is_feature_iterative_refinement(self):
        """When enabled, the model performs multiple iterations of entity/relation
        refinement using fact representations and multi-scale features."""
        return self.features & (1 << 24) != 0

    @property
    @feature_info(25, "Enable multi-scale consistency loss")
    def is_feature_multiscale_consistency_loss(self):
        """When enabled, adds a consistency loss between different scale representations."""
        return self.features & (1 << 25) != 0

    @property
    @feature_info(26, "Enable gradient isolation between parallel branches (detach multiscale from fact)")
    def is_feature_gradient_isolation(self):
        """When enabled, prevents fact decomposition gradients from flowing into the
        multi-scale module by detaching the shared encoder output before the fact branch.
        Only effective when both modules are enabled (bits 22 + 23)."""
        return self.features & (1 << 26) != 0

    @property
    @feature_info(27, "Enable hierarchical long-context NER boundary refinement")
    def is_feature_long_context_ner_refiner(self):
        """Use local boundary cues plus masked chunk-level context for NER."""
        return self.features & (1 << 27) != 0

    def list_features(self, logger):
        for k, v in FeaturesMixin.__dict__.items():
            if not k.startswith("is_feature_"):
                continue
            active = getattr(self, k)
            bit, desc = feature_descriptions[k]
            if active:
                logger.info(f"Using feature '{desc}' (bit {bit}, {1 << bit})")
