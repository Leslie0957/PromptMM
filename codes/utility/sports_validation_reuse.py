"""Narrow reuse authorization for the audited Validation-only Sports teacher."""
from utility.dataset_profiles import (
    SPORTS_STUDENT_PROFILE_NAMES, resolved_profile_metadata,
    resolved_student_profile_metadata,
)
from utility.experiment_protocol import file_fingerprint


SPORTS_TEACHER_SHA256 = '57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea'


def is_pinned_sports_validation_reuse(args, checkpoint_path):
    """Authorize eligibility relaxation only; caller still validates all metadata."""
    name = getattr(args, 'student_profile', '')
    if args.dataset != 'sports' or name not in SPORTS_STUDENT_PROFILE_NAMES:
        return False
    if (args.run_final_test or args.if_train_teacher or args.teacher_only
            or args.run_efficiency_benchmark or args.smoke_train_batches != 0
            or args.eval_protocol != 'val_test_once_v1'):
        raise ValueError('Sports diagnostic teacher reuse requires Validation-only student training.')
    if (resolved_profile_metadata('sports', args)['overrides']
            or resolved_student_profile_metadata('sports', name, args)['overrides']):
        raise ValueError('Sports diagnostic teacher reuse rejects profile overrides.')
    if file_fingerprint(checkpoint_path)['sha256'] != SPORTS_TEACHER_SHA256:
        raise ValueError('Sports diagnostic teacher checkpoint fingerprint mismatch.')
    return True
