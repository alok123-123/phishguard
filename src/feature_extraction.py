"""
feature_extraction.py — Extract structural URL features for live prediction.

This module extracts features from a raw URL string using ONLY the URL text.
It does NOT visit the website, download content, or execute any code from the URL.

Features extracted (structural, URL-only):
    1.  url_length           — Total characters in the URL
    2.  hostname_length      — Characters in the hostname
    3.  path_length          — Characters in the URL path
    4.  num_dots             — Number of '.' in the hostname
    5.  num_hyphens          — Number of '-' in the URL
    6.  num_underscores      — Number of '_' in the URL
    7.  num_slashes          — Number of '/' in the URL path
    8.  num_question_marks   — Number of '?' in the URL
    9.  num_equals           — Number of '=' in the URL
    10. num_at_signs         — Number of '@' in the URL
    11. num_ampersands       — Number of '&' in the URL
    12. num_digits_in_url    — Count of digit characters
    13. num_digits_in_host   — Count of digit characters in hostname
    14. num_subdomains       — Number of subdomain levels
    15. has_ip_address       — Whether hostname looks like an IP address
    16. has_https            — Whether the scheme is HTTPS
    17. has_http             — Whether the scheme is HTTP (not HTTPS)
    18. has_at_sign          — Binary: '@' present in URL
    19. has_double_slash_redirect — '//' appears in path (potential redirect)
    20. is_shortened         — Whether the domain is a known URL shortener
    21. path_depth           — Number of '/' separated path segments
    22. query_length         — Length of the query string
    23. fragment_length      — Length of the fragment (after '#')
    24. has_port             — Whether an explicit port number is specified
    25. special_char_count   — Total special characters in URL
    26. letter_ratio         — Ratio of letters to total URL length
    27. digit_ratio          — Ratio of digits to total URL length
    28. has_suspicious_tld   — Whether the TLD is commonly abused

IMPORTANT LIMITATION:
    The primary dataset (Hannousse & Yahiouche 2021) contains 87 features
    that include HTML content-based and external service features (e.g.,
    Google index, page rank, DNS record, web traffic). These features CANNOT
    be extracted from the URL string alone.

    This module extracts the URL-structural subset only. The training pipeline
    uses all features from the dataset, but live URL prediction uses only
    the features this module can extract. A separate "URL-only" model is
    trained on this reduced feature set for live predictions.

Usage:
    from src.feature_extraction import extract_url_features
    features = extract_url_features("https://example.com/login?user=test")
"""

import re
import json
import os
from urllib.parse import urlparse, parse_qs

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Suspicious TLDs (commonly abused, documented in anti-phishing research) ───
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",  # Freenom free TLDs
    ".buzz", ".top", ".xyz", ".club", ".work", ".info",
    ".site", ".online", ".icu", ".cam", ".rest",
}


def validate_url(url: str) -> str:
    """
    Validate and normalize a URL string.

    - Strips whitespace.
    - Rejects URLs that are too long (security measure).
    - Adds 'http://' if no scheme is provided (for parsing only).
    - Rejects empty or obviously malformed input.

    Parameters
    ----------
    url : str
        Raw URL string from user input.

    Returns
    -------
    str
        Normalized URL string.

    Raises
    ------
    ValueError
        If the URL is invalid or too long.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    url = url.strip()

    if len(url) > config.MAX_URL_LENGTH:
        raise ValueError(
            f"URL is too long ({len(url)} characters). "
            f"Maximum allowed: {config.MAX_URL_LENGTH}."
        )

    # Add scheme if missing (for proper urlparse behavior)
    if not url.startswith(("http://", "https://", "ftp://")):
        url = "http://" + url

    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"Could not extract a valid hostname from: {url}")

    return url


def _is_ip_address(hostname: str) -> bool:
    """Check if the hostname looks like an IP address (IPv4)."""
    # IPv4 pattern: four groups of 1-3 digits separated by dots
    ipv4_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    return bool(re.match(ipv4_pattern, hostname))


def _count_subdomains(hostname: str) -> int:
    """
    Count subdomain levels.
    Example: "mail.login.example.com" → 2 subdomains (mail, login)
    We consider the last two parts as domain + TLD.
    """
    parts = hostname.split(".")
    if len(parts) <= 2:
        return 0
    # For IPs, no subdomains
    if _is_ip_address(hostname):
        return 0
    return len(parts) - 2


def _get_tld(hostname: str) -> str:
    """Extract the top-level domain from the hostname."""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return "." + parts[-1]
    return ""


def extract_url_features(url: str) -> dict:
    """
    Extract structural features from a single URL string.

    This function ONLY parses the URL text. It does NOT:
        - Visit the website
        - Download any content
        - Execute JavaScript
        - Make DNS queries
        - Contact external APIs

    Parameters
    ----------
    url : str
        A validated URL string.

    Returns
    -------
    dict
        Dictionary of feature_name → feature_value.
    """
    # Validate and normalize
    url = validate_url(url)
    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    features = {}

    # ─── Length-based features ───
    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)
    features["query_length"] = len(query)
    features["fragment_length"] = len(fragment)

    # ─── Character count features ───
    features["num_dots"] = hostname.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_slashes"] = path.count("/")
    features["num_question_marks"] = url.count("?")
    features["num_equals"] = url.count("=")
    features["num_at_signs"] = url.count("@")
    features["num_ampersands"] = url.count("&")
    features["num_digits_in_url"] = sum(c.isdigit() for c in url)
    features["num_digits_in_host"] = sum(c.isdigit() for c in hostname)

    # ─── Structural features ───
    features["num_subdomains"] = _count_subdomains(hostname)
    features["path_depth"] = len([seg for seg in path.split("/") if seg])
    features["has_ip_address"] = int(_is_ip_address(hostname))
    features["has_https"] = int(parsed.scheme.lower() == "https")
    features["has_http"] = int(parsed.scheme.lower() == "http")
    features["has_at_sign"] = int("@" in url)
    features["has_double_slash_redirect"] = int("//" in path)
    features["is_shortened"] = int(hostname.lower() in config.URL_SHORTENERS)
    features["has_port"] = int(parsed.port is not None and parsed.port not in (80, 443))

    # ─── Ratio features ───
    url_len = max(len(url), 1)  # Avoid division by zero
    features["letter_ratio"] = round(
        sum(c.isalpha() for c in url) / url_len, 4
    )
    features["digit_ratio"] = round(
        sum(c.isdigit() for c in url) / url_len, 4
    )

    # ─── Special characters ───
    special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?~`")
    features["special_char_count"] = sum(1 for c in url if c in special_chars)

    # ─── Suspicious TLD ───
    tld = _get_tld(hostname)
    features["has_suspicious_tld"] = int(tld.lower() in SUSPICIOUS_TLDS)

    return features


def get_feature_names() -> list:
    """
    Return the ordered list of URL-structural feature names.

    This defines the schema for the URL-only model. The order MUST match
    what the model was trained on.

    Returns
    -------
    list of str
        Feature names in the correct order.
    """
    # Extract features from a dummy URL to get the key order
    dummy_features = extract_url_features("http://example.com")
    return list(dummy_features.keys())


def features_to_dataframe(features: dict):
    """
    Convert a feature dictionary to a single-row pandas DataFrame.

    The column order matches the schema from get_feature_names().

    Parameters
    ----------
    features : dict
        Feature dictionary from extract_url_features().

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame ready for model input.
    """
    import pandas as pd

    expected_names = get_feature_names()
    row = {}

    for name in expected_names:
        if name not in features:
            raise KeyError(
                f"Missing feature '{name}'. The feature extractor did not "
                f"produce all required features."
            )
        row[name] = features[name]

    return pd.DataFrame([row])


def compare_with_dataset_features(dataset_features: list) -> dict:
    """
    Compare URL-extractable features with the full dataset feature set.

    This documents which dataset features can and cannot be reproduced
    from a URL alone.

    Parameters
    ----------
    dataset_features : list
        Column names from the training dataset.

    Returns
    -------
    dict
        Report with 'extractable', 'unavailable', and 'extra' feature lists.
    """
    url_features = set(get_feature_names())
    dataset_set = set(dataset_features)

    report = {
        "url_extractable": sorted(url_features),
        "in_dataset_only": sorted(dataset_set - url_features),
        "in_url_only": sorted(url_features - dataset_set),
        "shared": sorted(url_features & dataset_set),
        "n_url_features": len(url_features),
        "n_dataset_features": len(dataset_set),
        "n_shared": len(url_features & dataset_set),
    }

    print("\n[feature_extraction] Feature Comparison Report:")
    print(f"  URL-extractable features: {report['n_url_features']}")
    print(f"  Dataset features:         {report['n_dataset_features']}")
    print(f"  Shared features:          {report['n_shared']}")
    print(f"  Dataset-only features:    {len(report['in_dataset_only'])}")
    print(f"  URL-only features:        {len(report['in_url_only'])}")

    return report
