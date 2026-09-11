"""Redact credentials from configuration metadata and diagnostic messages."""

import re


_SECRET_KEYS = {
    'apikey', 'apitoken', 'accesstoken', 'authtoken', 'bearertoken', 'token',
    'authorization', 'password', 'passwd', 'secret', 'clientsecret', 'credential',
}
_TOKEN_PATTERNS = (
    re.compile(r'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}'),
    re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),
    re.compile(r'AIza[0-9A-Za-z_-]{30,}'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'(?i)\bBearer\s+[^\s,;\'\"}]+'),
)


def redact_text(text, *, secrets=()):
    """Mask known credentials and common token forms without logging them."""
    result = str(text)
    for secret in secrets:
        if secret:
            result = result.replace(str(secret), '[REDACTED]')
    for pattern in _TOKEN_PATTERNS:
        result = pattern.sub('[REDACTED]', result)
    return re.sub(r'(https?://)[^/\s@]+@', r'\1[REDACTED]@', result)


def redact_metadata(value, *, secrets=()):
    """Return a redacted copy; never alter the configuration sent to the API."""
    if isinstance(value, dict):
        return {
            key: ('[REDACTED]' if re.sub(r'[^a-z]', '', str(key).lower()) in _SECRET_KEYS
                  else redact_metadata(item, secrets=secrets))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_metadata(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    return value
