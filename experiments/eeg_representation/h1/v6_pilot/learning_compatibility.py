"""Narrow compatibility for completed pre-recovery ridge artifacts.

Only the exact approved TCN metadata/signature patch may retain the original
ridge source identity. Inputs, risk contracts, settings, and every saved output
checksum still pass through the original ridge resume validator. New fits use
the current source identity. No old TCN completion is accepted by this adapter.
"""
from pathlib import Path
import hashlib
import inspect

from . import learning

LEGACY_COMMIT = '9e8740635f7585066c1e6903999c022bf634003a'
LEGACY_SOURCE_SHA256 = 'fa33f15add878b00ec7730029a12d64088af7250d39b0ee50b468f6003dc6314'

# Each old/new pair was checked against git show LEGACY_COMMIT:.../learning.py.
# Reverse exactly these edits and hash the complete file, including whitespace.
# Any other source change therefore blocks the old-identity exception.
_APPROVED_PATCHES = (
    ("def signature(panel,values,manifest,settings):",
     "def signature(panel,values,manifest,settings,*,implementation_sha256=None):"),
    ("risk_contract=manifest['risk_contract'],settings=settings,implementation=file_hash(Path(__file__))))",
     "risk_contract=manifest['risk_contract'],settings=settings,implementation=file_hash(Path(__file__)) if implementation_sha256 is None else implementation_sha256))"),
    ("def ridge(panel,values,manifest,destination,config,*,resume=False,mechanism_scaled=False,design=None):",
     "def ridge(panel,values,manifest,destination,config,*,resume=False,mechanism_scaled=False,design=None,signature_implementation_sha256=None):"),
    ("    sig=signature(panel,values,manifest,settings)",
     "    sig=signature(panel,values,manifest,settings,implementation_sha256=signature_implementation_sha256)"),
    ("    result=dict(status='complete',signature=sig,selected_id='tcn' if use else 'train_mean',history_samples=history,",
     "    result=dict(account,status='complete',signature=sig,selected_id='tcn' if use else 'train_mean',history_samples=history,"),
    ("        selection_split='development',mean_fallback_retained=True,**account,",
     "        selection_split='development',mean_fallback_retained=True,"),
)


def compatibility_record(source_path=None):
    """Validate the complete current file against the one approved text patch."""
    path = Path(learning.__file__ if source_path is None else source_path)
    current = path.read_bytes()
    restored = current
    for old, new in _APPROVED_PATCHES:
        encoded = new.encode('utf-8')
        if restored.count(encoded) != 1:
            raise ValueError('Unapproved learner source: exact compatibility patch absent or ambiguous')
        restored = restored.replace(encoded, old.encode('utf-8'), 1)
    if hashlib.sha256(restored).hexdigest() != LEGACY_SOURCE_SHA256:
        raise ValueError('Unapproved learner source: changes beyond the frozen compatibility patch')
    return dict(status='verified', legacy_commit=LEGACY_COMMIT,
        legacy_learning_sha256=LEGACY_SOURCE_SHA256,
        current_learning_sha256=hashlib.sha256(current).hexdigest(),
        patch_scope='TCN completion metadata duplicate-key fix and explicit ridge signature identity override only; no ridge numerical change',
        reuse_scope='Existing completed ridge selection records only; all input/settings/risk-contract and output checksum checks remain mandatory',
        new_fit_identity='current learning.py SHA256', tcn_legacy_resume=False)


def compatible_ridge(original, *args, **kwargs):
    """Call regular ridge, allowing only verified old-source resume.

    No filesystem mutation occurs on either resume branch. A legacy signature
    is retried only after the ordinary validator reports its exact identity
    mismatch; the retry independently rejects changed inputs or settings.
    """
    if kwargs.get('signature_implementation_sha256') is not None:
        raise ValueError('The compatibility wrapper owns the signature identity override')
    try:
        return original(*args, **kwargs)
    except ValueError as exc:
        if str(exc) != 'Pilot readout inputs/settings changed':
            raise
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments['resume']:
            raise
        marker = Path(bound.arguments['destination']) / 'selection.json'
        if not marker.is_file():
            raise
        compatibility_record()
        return original(*args, **dict(kwargs, signature_implementation_sha256=LEGACY_SOURCE_SHA256))
