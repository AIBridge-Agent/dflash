# pyright: reportMissingImports=none
import pytest

from dflash.residual_errors import (
    ConfigurationError,
    ExactnessViolation,
    FairnessViolation,
    InvalidAccountingError,
    InvalidBudgetError,
    InvalidCandidateError,
    InvalidRerootError,
    NoResidualOpportunity,
    ProvenanceViolation,
    RemoteExecutionError,
    ResidualDFlashError,
)


def test_residual_errors_share_common_root() -> None:
    error_types = [
        InvalidCandidateError,
        NoResidualOpportunity,
        InvalidRerootError,
        ProvenanceViolation,
        InvalidBudgetError,
        InvalidAccountingError,
        ExactnessViolation,
        ConfigurationError,
        RemoteExecutionError,
        FairnessViolation,
    ]

    for error_type in error_types:
        assert issubclass(error_type, ResidualDFlashError)


def test_residual_error_message_is_preserved() -> None:
    with pytest.raises(InvalidCandidateError, match="bad probability"):
        raise InvalidCandidateError("bad probability")
