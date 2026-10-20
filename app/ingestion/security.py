"""SSRF protection for ingested URLs."""

import ipaddress
import socket
import urllib.parse


class SSRFBlockedError(ValueError):
    """Raised when a URL resolves to a private, loopback, link-local, or reserved address."""


def _is_restricted_ip(ip_obj: ipaddress._BaseAddress) -> bool:
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def validate_url_against_ssrf(url: str) -> str:
    """
    Validates that `url` uses http/https and does not target localhost, private LAN ranges,
    loopback (127.0.0.0/8, ::1), or cloud metadata link-local addresses (169.254.169.254).
    """
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise SSRFBlockedError(
            f"Blocked URL scheme '{parsed.scheme}'. Only http and https are allowed."
        )

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise SSRFBlockedError("URL is missing a valid hostname.")

    if hostname in ("localhost", "localhost.localdomain") or hostname.endswith(
        (".local", ".internal", ".localhost")
    ):
        raise SSRFBlockedError(f"Blocked local/internal hostname: {hostname}")

    # Check literal IP addresses before resolving DNS.
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if _is_restricted_ip(ip_obj):
            raise SSRFBlockedError(f"Blocked restricted IP address: {ip_obj}")
        return url.strip()
    except ValueError as error:
        if isinstance(error, SSRFBlockedError):
            raise

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname '{hostname}': {exc}") from exc

    if not addr_info:
        raise ValueError(f"No DNS records found for hostname '{hostname}'.")

    for _, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        # Strip IPv6 scope ID if present.
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        ip_obj = ipaddress.ip_address(ip_str)
        # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1).
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
            ip_obj = ip_obj.ipv4_mapped
        if _is_restricted_ip(ip_obj):
            raise SSRFBlockedError(
                f"Blocked SSRF attempt: '{hostname}' resolved to restricted IP {ip_obj}."
            )

    return url.strip()
