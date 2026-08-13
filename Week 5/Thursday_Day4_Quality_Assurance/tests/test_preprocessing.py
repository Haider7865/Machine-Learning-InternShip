"""Task 10 — Preprocessing Testing
Every preprocessing function is tested with normal, missing, invalid, and
boundary inputs, as required by the module brief."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import validate_customer_input, ValidationError


# ---------------- Normal input ----------------

def test_valid_normal_input():
    result = validate_customer_input({
        "age": 35, "income": 50000, "total_spending": 1200,
        "web_purchases": 5, "store_purchases": 8, "recency": 20,
    })
    assert result["age"] == 35
    assert result["income"] == 50000


# ---------------- Missing input ----------------

def test_missing_required_field():
    with pytest.raises(ValidationError, match="required"):
        validate_customer_input({
            "age": 35, "income": 50000, "total_spending": 1200,
            "web_purchases": 5, "store_purchases": 8,
            # "recency" missing
        })


def test_empty_string_field():
    with pytest.raises(ValidationError, match="required"):
        validate_customer_input({
            "age": "", "income": 50000, "total_spending": 1200,
            "web_purchases": 5, "store_purchases": 8, "recency": 20,
        })


# ---------------- Invalid input ----------------

def test_negative_income():
    with pytest.raises(ValidationError, match="negative"):
        validate_customer_input({
            "age": 35, "income": -5000, "total_spending": 1200,
            "web_purchases": 5, "store_purchases": 8, "recency": 20,
        })


def test_negative_spending():
    with pytest.raises(ValidationError, match="negative"):
        validate_customer_input({
            "age": 35, "income": 50000, "total_spending": -500,
            "web_purchases": 5, "store_purchases": 8, "recency": 20,
        })


def test_negative_age():
    with pytest.raises(ValidationError, match="between 0 and 120"):
        validate_customer_input({
            "age": -10, "income": 50000, "total_spending": 1200,
            "web_purchases": 5, "store_purchases": 8, "recency": 20,
        })


def test_text_in_numeric_field():
    with pytest.raises(ValidationError, match="must be a number"):
        validate_customer_input({
            "age": "abc", "income": 50000, "total_spending": 1200,
            "web_purchases": 5, "store_purchases": 8, "recency": 20,
        })


def test_extremely_large_income():
    with pytest.raises(ValidationError, match="unrealistically large"):
        validate_customer_input({
            "age": 35, "income": 999_999_999, "total_spending": 1200,
            "web_purchases": 5, "store_purchases": 8, "recency": 20,
        })


# ---------------- Boundary values ----------------

def test_boundary_age_zero():
    result = validate_customer_input({
        "age": 0, "income": 50000, "total_spending": 1200,
        "web_purchases": 5, "store_purchases": 8, "recency": 20,
    })
    assert result["age"] == 0


def test_boundary_age_120():
    result = validate_customer_input({
        "age": 120, "income": 50000, "total_spending": 1200,
        "web_purchases": 5, "store_purchases": 8, "recency": 20,
    })
    assert result["age"] == 120


def test_boundary_age_121_rejected():
    with pytest.raises(ValidationError):
        validate_customer_input({
            "age": 121, "income": 50000, "total_spending": 1200,
            "web_purchases": 5, "store_purchases": 8, "recency": 20,
        })


def test_boundary_income_zero():
    result = validate_customer_input({
        "age": 35, "income": 0, "total_spending": 0,
        "web_purchases": 0, "store_purchases": 0, "recency": 0,
    })
    assert result["income"] == 0
