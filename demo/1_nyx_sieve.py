
import re
import hmac
import hashlib
from datetime import date, datetime
import secrets


# DEMO ONLY: use a secret stored outside the source code in production.
SECRET_KEY = bytes.fromhex(secrets.token_hex(32))

# Stable mappings for this document.
name_map = {}
provider_map = {}
facility_map = {}
id_map = {}

# The first admission date is the temporal reference point.
ADMISSION_DATE = date(2026, 9, 28)


def stable_token(value: str, prefix: str, length: int = 8) -> str:
    """Create a deterministic pseudonymous token."""
    digest = hmac.new(
        SECRET_KEY,
        value.strip().casefold().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:length]
    return f"{prefix}-{digest}"


def mapped_token(value: str, mapping: dict, prefix: str) -> str:
    """Return the same sequential token for repeated values in this run."""
    key = value.strip().casefold()
    if key not in mapping:
        mapping[key] = f"{prefix}-{len(mapping) + 1:02d}"
    return mapping[key]


def relative_date(value: str) -> str:
    """Convert YYYY-MM-DD to a relative day from admission."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return value

    offset = (parsed - ADMISSION_DATE).days
    return "Day 0" if offset == 0 else (
        f"Day +{offset}" if offset > 0 else f"Day {offset}"
    )


def anonymize(note: str) -> str:
    # Names: example expects labeled fields in this input format.
    note = re.sub(
        r"(?im)^(Patient|Patient Name):\s*(.+)$",
        lambda m: f"{m.group(1)}: "
        + mapped_token(m.group(2), name_map, "Subject"),
        note,
    )

    # Provider and facility names.
    note = re.sub(
        r"(?im)^(Provider|Doctor):\s*(.+)$",
        lambda m: f"{m.group(1)}: "
        + mapped_token(m.group(2), provider_map, "Provider"),
        note,
    )
    note = re.sub(
        r"(?im)^(Facility|Hospital|Clinic):\s*(.+)$",
        lambda m: f"{m.group(1)}: "
        + mapped_token(m.group(2), facility_map, "Facility"),
        note,
    )

    # Dates in labeled fields: preserve relative time relationships.
    def replace_date_field(match):
        label, value = match.group(1), match.group(2)
        return f"{label}: {relative_date(value)}"

    note = re.sub(
        r"(?im)^(Admitted on|Visit Date|Discharged on):\s*"
        r"(\d{4}-\d{2}-\d{2})\s*$",
        replace_date_field,
        note,
    )

    # DOB: replace with age in years at the admission date.
    def replace_dob(match):
        dob = date.fromisoformat(match.group(1))
        age = ADMISSION_DATE.year - dob.year
        if (ADMISSION_DATE.month, ADMISSION_DATE.day) < (
            dob.month, dob.day
        ):
            age -= 1
        return f"Age: {age}y"

    note = re.sub(
        r"(?im)^(?:DOB|Date of Birth):\s*(\d{4}-\d{2}-\d{2})\s*$",
        replace_dob,
        note,
    )

    # IDs: HMAC avoids exposing the original ID in the token.
    def replace_id(match):
        label, value = match.group(1), match.group(2)
        return f"{label}: {stable_token(value, 'ID-HASH')}"

    note = re.sub(
        r"(?im)^(MRN|SSN|National ID|Clinic ID):\s*(.+)$",
        replace_id,
        note,
    )

    return note


if __name__ == "__main__":
    # Synthetic sample — do not use real patient information.
    raw_note = """Patient Name: Johnathan Miller
DOB: 1968-04-12
SSN: 994821
MRN: 123456
Provider: Dr. Sarah Chen    
Facility: Mercy General
Admitted on: 2026-09-28
Visit Date: 2026-09-29
Symptoms: Headache and dizziness for three days.
"""

    sanitized_note = anonymize(raw_note)

    print("=" * 24 + " RAW EHR INPUT " + "=" * 24)
    print(raw_note)
    print("=" * 20 + " SANITIZED NYX SIEVE PAYLOAD " + "=" * 20)
    print(sanitized_note)
