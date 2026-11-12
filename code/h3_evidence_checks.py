"""Failure-persisting checks for a future, independently admitted H3 run.

The frozen joint probe's ``require(value, message, evidence)`` saves the full
report on every call. This separate helper keeps its Python truth test and
RuntimeError(message) rejection, but successful checks neither mutate evidence
nor save it. The caller must still persist explicit phase/checkpoint progress.

Rejected checks append an additive ``gate_failures`` record before saving and
raising. Truth-test errors are saved and re-raised unchanged. A save error is
not swallowed: as in the old helper, it takes precedence over gate rejection.
This module does not patch or resume any old process, import a model, or make a
previous failed run successful.
"""


def _record_failure(message, evidence, evaluation_error=None):
    record = {"message": message, "phase": evidence.data.get("phase")}
    if evaluation_error is not None:
        record["evaluation_error"] = {
            "type": type(evaluation_error).__name__,
            "message": str(evaluation_error),
        }
    evidence.data.setdefault("gate_failures", []).append(record)
    evidence.save()


def require(value, message, evidence):
    """Return None on truth; persist failure before the original exception.

    ``value`` is tested exactly once with normal Python truthiness, not coerced
    to a stricter boolean type. ``evidence`` has the existing ``data`` dict and
    ``save()`` interface. Successful checks require neither interface member.
    """
    try:
        if value:
            return
    except BaseException as error:
        _record_failure(message, evidence, evaluation_error=error)
        raise
    _record_failure(message, evidence)
    raise RuntimeError(message)
