"""Failure classes that decide where a job goes when processing fails."""


class PermanentError(Exception):
    """Retrying will not help (corrupt file, unsupported type, no text). Goes straight to the DLQ."""


class TransientError(Exception):
    """May succeed later (network blip, overloaded service). Retried with backoff."""


class RateLimitError(TransientError):
    """The embedding provider asked us to slow down."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(Exception):
    """Misconfiguration (e.g. bad API key). The worker stops without committing, so no jobs are lost."""
