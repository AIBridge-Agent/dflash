# pyright: reportMissingImports=none
"""Domain exceptions for zero-draft residual verification."""


class ResidualDFlashError(Exception):
    """Base class for residual DFlash errors."""


class InvalidCandidateError(ResidualDFlashError):
    """Raised when a residual candidate or path is malformed."""


class NoResidualOpportunity(ResidualDFlashError):
    """Raised when early rejection leaves no residual tail to verify."""


class InvalidRerootError(ResidualDFlashError):
    """Raised when residual re-rooting receives inconsistent inputs."""


class ProvenanceViolation(ResidualDFlashError):
    """Raised when residual output cannot be traced to the original draft tail."""


class InvalidBudgetError(ResidualDFlashError):
    """Raised when a residual DDTree budget is invalid."""


class InvalidAccountingError(ResidualDFlashError):
    """Raised when throughput or verification accounting is invalid."""


class ExactnessViolation(ResidualDFlashError):
    """Raised when a residual token would be accepted without target verification."""


class ConfigurationError(ResidualDFlashError):
    """Raised when residual runtime configuration is invalid."""


class RemoteExecutionError(ResidualDFlashError):
    """Raised when A100 remote execution would violate safety constraints."""


class FairnessViolation(ResidualDFlashError):
    """Raised when a pilot report cannot support fair wall-clock claims."""
