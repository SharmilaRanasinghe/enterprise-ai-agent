
ALLOWED_ROLES = {"viewer", "engineer", "admin"}


def can_access_document(
    document: dict,
    role: str,
) -> bool:

    if role not in ALLOWED_ROLES:
        return False

    level = document.get("access_level")
    department = document.get("department")

    if level not in {"public", "internal", "restricted"}:
        return False

    if role == "admin":
        return True

    if level == "public":
        return True

    return (
        role == "engineer"
        and level == "internal"
        and department == "Engineering"
    )


def allowed_document_filter(role: str) -> dict:

    if role == "admin":
        return {
            "access_level": {
                "$in": ["public", "internal", "restricted"]
            }
        }

    if role == "engineer":
        return {
            "$or": [
                {
                    "access_level": {
                        "$eq": "public"
                    }
                },
                {
                    "$and": [
                        {
                            "access_level": {
                                "$eq": "internal"
                            }
                        },
                        {
                            "department": {
                                "$eq": "Engineering"
                            }
                        },
                    ]
                },
            ]
        }

    if role == "viewer":
        return {
            "access_level": {
                "$eq": "public"
            }
        }

    raise ValueError("Unknown role")
