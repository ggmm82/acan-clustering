"""Named CPU checks required before publishing controlled LM tooling."""
PROTOCOL='controlled-v1'
MIN_TESTS=26
REQUIRED_TESTS={
    'test_layer_inspection_preserves_exact_training_and_is_owner_scoped',
    'test_layer_inspection_tracks_prefill_decoding_and_does_not_change_logits',
    'test_distillation_resume_preserves_exact_parameters',
    'test_document_deduplication_and_pause_recovery_do_not_cross_splits',
    'test_early_stop_preserves_best_after_rolling_prune_and_can_resume',
    'test_export_contains_inspectable_deltas_and_reloads_exact_weights',
    'test_full_acan_is_causal_streaming_and_uses_no_dense_kv_cache',
    'test_independent_test_comparison_measures_both_quality_and_whole_cache',
    'test_web_diagnostics_and_export_are_owner_scoped_and_redact_secrets',
}
