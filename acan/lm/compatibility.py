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
PRE_JOINT_TRAINING = '8acbb6907ceb994e7963a9008b7ada33840a00b2c83f130d7d4e43baef336282'
PRE_JOINT_INFERENCE = 'cbbb435f05f9f757587b483918039b802dc79ddd4c26d7d4095ec6d9ba68023a'
# Sources immediately before the cp-v9 attention profile was added. Projects of
# every earlier profile keep their trainer and inference equations unchanged.
PRE_V9_TRAINING = '0516a223d3601f35e8cae79453948217937366a241bd2c1858c6272bd17e5c66'
PRE_V9_INFERENCE = 'fdd4c2cbc515abf90feeb2a7a96f99f9f413ce8561c00344c2f03c920fe4215f'


def training_identity(values, config, checkpoint_root, selected, current):
    from pathlib import Path
    from . import checkpoints
    candidates = [current]
    if config.get('routing_profile') != 'cp-v9':
        candidates.append(PRE_V9_TRAINING)
    if 'cp_weight' not in config and config.get('routing_profile') in ('shared-algebra-v1','legacy-causal-v1','cp-routing-v1'):
        candidates.append(PRE_JOINT_TRAINING)
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
