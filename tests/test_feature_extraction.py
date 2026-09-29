"""
test_feature_extraction.py — Tests for URL feature extraction.

Tests cover:
    - Valid URL feature extraction
    - Invalid URL handling
    - Edge cases (IP addresses, URL shorteners, special characters)
    - Feature schema consistency
    - URL validation logic
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_extraction import (
    extract_url_features,
    validate_url,
    get_feature_names,
    features_to_dataframe,
    compare_with_dataset_features,
)


class TestValidateUrl:
    """Tests for URL validation."""

    def test_valid_https_url(self):
        result = validate_url("https://example.com")
        assert result == "https://example.com"

    def test_valid_http_url(self):
        result = validate_url("http://example.com")
        assert result == "http://example.com"

    def test_adds_scheme_if_missing(self):
        result = validate_url("example.com")
        assert result == "http://example.com"

    def test_strips_whitespace(self):
        result = validate_url("  https://example.com  ")
        assert result == "https://example.com"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_url("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_url(None)

    def test_too_long_url_raises(self):
        long_url = "https://example.com/" + "a" * 3000
        with pytest.raises(ValueError, match="too long"):
            validate_url(long_url)

    def test_no_hostname_raises(self):
        with pytest.raises(ValueError):
            validate_url("http://")


class TestExtractUrlFeatures:
    """Tests for feature extraction from URLs."""

    def test_basic_url_returns_dict(self):
        features = extract_url_features("https://example.com")
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_url_length_feature(self):
        url = "https://example.com"
        features = extract_url_features(url)
        assert features["url_length"] == len(url)

    def test_hostname_length(self):
        features = extract_url_features("https://example.com/path")
        assert features["hostname_length"] == len("example.com")

    def test_https_detection(self):
        features = extract_url_features("https://example.com")
        assert features["has_https"] == 1
        assert features["has_http"] == 0

    def test_http_detection(self):
        features = extract_url_features("http://example.com")
        assert features["has_https"] == 0
        assert features["has_http"] == 1

    def test_ip_address_detection(self):
        features = extract_url_features("http://192.168.1.1/login")
        assert features["has_ip_address"] == 1

    def test_no_ip_address(self):
        features = extract_url_features("https://google.com")
        assert features["has_ip_address"] == 0

    def test_at_sign_detection(self):
        features = extract_url_features("https://user@evil.com")
        assert features["has_at_sign"] == 1
        assert features["num_at_signs"] >= 1

    def test_no_at_sign(self):
        features = extract_url_features("https://example.com")
        assert features["has_at_sign"] == 0

    def test_subdomain_count(self):
        features = extract_url_features("https://sub1.sub2.example.com")
        assert features["num_subdomains"] == 2

    def test_no_subdomains(self):
        features = extract_url_features("https://example.com")
        assert features["num_subdomains"] == 0

    def test_url_shortener_detection(self):
        features = extract_url_features("https://bit.ly/abc123")
        assert features["is_shortened"] == 1

    def test_not_shortened(self):
        features = extract_url_features("https://example.com/page")
        assert features["is_shortened"] == 0

    def test_special_characters(self):
        features = extract_url_features("https://example.com/page?a=1&b=2#section")
        assert features["num_question_marks"] >= 1
        assert features["num_equals"] >= 1
        assert features["num_ampersands"] >= 1

    def test_dots_in_hostname(self):
        features = extract_url_features("https://www.example.co.uk")
        assert features["num_dots"] >= 3

    def test_query_length(self):
        features = extract_url_features("https://example.com/page?key=value")
        assert features["query_length"] > 0

    def test_fragment_length(self):
        features = extract_url_features("https://example.com/page#section")
        assert features["fragment_length"] > 0

    def test_suspicious_tld(self):
        features = extract_url_features("https://malicious.tk")
        assert features["has_suspicious_tld"] == 1

    def test_normal_tld(self):
        features = extract_url_features("https://example.com")
        assert features["has_suspicious_tld"] == 0

    def test_letter_and_digit_ratios(self):
        features = extract_url_features("https://example123.com")
        assert 0 <= features["letter_ratio"] <= 1
        assert 0 <= features["digit_ratio"] <= 1

    def test_path_depth(self):
        features = extract_url_features("https://example.com/a/b/c")
        assert features["path_depth"] == 3

    def test_port_detection(self):
        features = extract_url_features("http://example.com:8080/page")
        assert features["has_port"] == 1

    def test_no_explicit_port(self):
        features = extract_url_features("https://example.com")
        assert features["has_port"] == 0


class TestFeatureSchema:
    """Tests for feature schema consistency."""

    def test_feature_names_returns_list(self):
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_all_features_present(self):
        """All expected features should be in the schema."""
        names = get_feature_names()
        expected = [
            "url_length", "hostname_length", "num_dots",
            "has_https", "has_ip_address", "is_shortened",
        ]
        for feat in expected:
            assert feat in names, f"Feature '{feat}' missing from schema"

    def test_feature_order_consistent(self):
        """Feature names should be the same each time."""
        names1 = get_feature_names()
        names2 = get_feature_names()
        assert names1 == names2

    def test_features_to_dataframe(self):
        """features_to_dataframe should produce correct column order."""
        features = extract_url_features("https://example.com")
        df = features_to_dataframe(features)
        assert list(df.columns) == get_feature_names()
        assert df.shape == (1, len(get_feature_names()))

    def test_compare_with_dataset_features(self):
        """Feature comparison should produce a valid report."""
        dataset_feats = ["url_length", "hostname_length", "page_rank", "dns_record"]
        report = compare_with_dataset_features(dataset_feats)
        assert "url_extractable" in report
        assert "in_dataset_only" in report
        assert "shared" in report
        assert "page_rank" in report["in_dataset_only"]


class TestEdgeCases:
    """Tests for edge cases and unusual URLs."""

    def test_long_path(self):
        url = "https://example.com/" + "/".join(["segment"] * 50)
        features = extract_url_features(url)
        assert features["path_depth"] == 50

    def test_many_query_params(self):
        params = "&".join([f"key{i}=val{i}" for i in range(20)])
        url = f"https://example.com/page?{params}"
        features = extract_url_features(url)
        assert features["num_ampersands"] >= 19

    def test_unicode_in_url(self):
        """Should handle URLs with unicode characters."""
        features = extract_url_features("https://example.com/日本語")
        assert isinstance(features, dict)

    def test_url_without_path(self):
        features = extract_url_features("https://example.com")
        assert features["path_length"] >= 0
