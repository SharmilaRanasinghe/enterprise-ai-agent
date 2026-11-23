
from backend.security.access import (
    can_access_document,
    allowed_document_filter,
)


def test_viewer_public_access():
    doc = {
        "access_level": "public",
        "department": "Engineering",
    }

    assert can_access_document(doc, "viewer")


def test_viewer_internal_denied():
    doc = {
        "access_level": "internal",
        "department": "Engineering",
    }

    assert not can_access_document(doc, "viewer")


def test_engineer_internal_engineering_allowed():
    doc = {
        "access_level": "internal",
        "department": "Engineering",
    }

    assert can_access_document(doc, "engineer")


def test_engineer_internal_compliance_denied():
    doc = {
        "access_level": "internal",
        "department": "Compliance",
    }

    assert not can_access_document(doc, "engineer")


def test_admin_restricted_allowed():
    doc = {
        "access_level": "restricted",
        "department": "Compliance",
    }

    assert can_access_document(doc, "admin")


def test_missing_classification_denied():
    doc = {
        "department": "Engineering",
    }

    assert not can_access_document(doc, "admin")


def test_unknown_role_denied():
    doc = {
        "access_level": "public",
        "department": "Engineering",
    }

    assert not can_access_document(doc, "unknown")


def test_viewer_pinecone_filter():
    assert allowed_document_filter("viewer") == {
        "access_level": {"$eq": "public"}
    }
