
import re
import hmac
import hashlib
from datetime import date, datetime
import secrets
from textwrap import wrap


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


def anonymize_entity(note, field_pattern, prefix, mapping):

    match = re.search(field_pattern, note, flags=re.IGNORECASE | re.MULTILINE)

    if not match:
        return note

    entity_name = match.group(1).strip()

    # Create a new ID if this entity hasn't been seen before
    if entity_name not in mapping:
        mapping[entity_name] = f"{prefix}-{len(mapping) + 1:02d}"

    anonymized_name = mapping[entity_name]

    # Replace the entity everywhere in the note
    note = re.sub(
        re.escape(entity_name),
        anonymized_name,
        note,
        flags=re.IGNORECASE
    )

    return note

def anonymize_dates(note):
    # Find admission date
    match = re.search(
        r"(?i)admitted on\s+(\d{4}-\d{2}-\d{2})",
        note
    )

    if not match:
        return note

    admission_date = datetime.strptime(
        match.group(1),
        "%Y-%m-%d"
    )

    # Find every YYYY-MM-DD date
    def replace_date(match):
        date_string = match.group(0)

        date = datetime.strptime(
            date_string,
            "%Y-%m-%d"
        )

        day_number = (date - admission_date).days

        if day_number == 0:
            return "Day 0"

        if day_number > 0:
            return f"Day {day_number}"

        return f"Day {day_number}"

    note = re.sub(
        r"\d{4}-\d{2}-\d{2}",
        replace_date,
        note
    )

    return note

def anonymize(note: str) -> str:
    subject_map = {}
    provider_map = {}
    facilitator_map = {}

    # Find the patient name
    note = anonymize_entity(
        note,
        r"(?im)^Patient(?: Name)?:\s*(.+)$",
        "Subject",
        subject_map
    )

    note = anonymize_entity(
        note,
        r"(?im)^(?:Provider|Doctor|Attending Physician):\s*(.+)$",
        "Provider",
        provider_map
    )

    note = anonymize_entity(
        note,
        r"(?i)^(?:Facility|Hospital|Clinic):\s*(.+)$",
        "Facility",
        facility_map
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

    
    note = anonymize_dates(note)

    return note

def format_side_by_side(left_text, right_text, width=55):
    left_lines = []
    right_lines = []

    # Preserve the blank lines
    for line in left_text.splitlines():
        left_lines.extend(wrap(line, width) or [""])

    for line in right_text.splitlines():
        right_lines.extend(wrap(line, width) or [""])

    max_lines = max(len(left_lines), len(right_lines))

    print(f"{'RAW EHR INPUT':<{width}} | {'SANITIZED NYX SIEVE PAYLOAD':<{width}}")
    print("-" * width + "-+-" + "-" * width)

    for i in range(max_lines):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""

        print(f"{left:<{width}} | {right:<{width}}")


if __name__ == "__main__":
    # Synthetic sample — do not use real patient information.
    # raw_note = """Patient Name: Johnathan Miller
    # DOB: 1968-04-12
    # SSN: 994821
    # MRN: 123456
    # Provider: Dr. Sarah Chen    
    # Facility: Mercy General
    # Admitted on: 2026-09-28
    # Visit Date: 2026-09-29
    # Symptoms: Headache and dizziness for three days.
    # """
    with open("sample_ehr.txt", "r", encoding="utf-8") as file:
        raw_note = file.read()

    sanitized_note = anonymize(raw_note)
    WIDTH = 55
    format_side_by_side(raw_note, sanitized_note)