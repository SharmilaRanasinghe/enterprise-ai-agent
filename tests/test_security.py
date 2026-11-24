
import pytest

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

@pytest.mark.parametrize(
    "role,access_level,department,expected",
    [
        ("viewer", "public", "Engineering", True),
        ("viewer", "internal", "Engineering", False),
        ("viewer", "restricted", "Engineering", False),

        ("engineer", "public", "Compliance", True),
        ("engineer", "internal", "Engineering", True),
        ("engineer", "internal", "Compliance", False),
        ("engineer", "restricted", "Engineering", False),

        ("admin", "public", "Engineering", True),
        ("admin", "internal", "Compliance", True),
        ("admin", "restricted", "Engineering", True),

        ("unknown", "public", "Engineering", False),
        ("admin", "unknown", "Engineering", False),
    ],
)
def test_document_access_control(
    role,
    access_level,
    department,
    expected,
):
    document = {
        "access_level": access_level,
        "department": department,
    }

    assert can_access_document(
        document,
        role,
    ) == expected


@pytest.mark.parametrize(
    "role",
    ["viewer", "engineer", "admin"],
)
def test_missing_access_metadata(role):

    document = {
        "department": "Engineering",
    }

    assert can_access_document(
        document,
        role,
    ) is False

from backend.main import secure_thread_id


def test_same_user_same_session():

    first = secure_thread_id(
        "user-001", "session-001"
    )

    second = secure_thread_id(
        "user-001", "session-001"
    )

    assert first == second


def test_different_users_same_session():

    first = secure_thread_id(
        "user-001", "session-001"
    )

    second = secure_thread_id(
        "user-002", "session-001"
    )

    assert first != second


def test_same_user_different_sessions():

    first = secure_thread_id(
        "user-001", "session-001"
    )

    second = secure_thread_id(
        "user-001", "session-002"
    )

    assert first != second