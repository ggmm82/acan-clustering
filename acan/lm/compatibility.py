"""Known controlled-v1 receipts whose numerical legacy trainer is unchanged.

These are hashes of the exact PR6 and pre-shared-algebra PR7 sources. Accepting a
receipt still requires the complete configuration/base/corpus/token fingerprint.
No shared-algebra project is allowed to opt into a legacy receipt.
"""
LEGACY_INFERENCE = '7894c97daaa56407e1f2fe50091552d835f21ebab1aa14dc10af428bc146a7a1'
LEGACY_TRAINING = (
    '00a1973c7bc31c81fa711ccc76b53bdde0e28bd6e6f70cb5a2aa3f814ca9729e',
    '104c751409871a0bed4416c32a764f9e02d30fd802d708e07a4608d5730139e2',
)


def training_identity(values, config, checkpoint_root, selected, current):
    from pathlib import Path
    from . import checkpoints
    candidates = [current]
    if 'routing_profile' not in config:
        candidates.extend(LEGACY_TRAINING)
    if config.get('routing_profile') in ('shared-algebra-v1','legacy-causal-v1'):
        candidates.append('c4abb615a710d213ef096ba2dc0600da59bda81f5dc458429198124a1e3c1462')
    if selected and Path(selected).name == selected and selected.startswith('step-'):
        path = checkpoint_root / selected
        if path.is_dir():
            previous = checkpoints.verify(path)['identity']
            for receipt in candidates:
                identity = checkpoints.fingerprint(dict(values, research_contract=receipt))
                if identity == previous:
                    return identity
    return checkpoints.fingerprint(dict(values, research_contract=current))
