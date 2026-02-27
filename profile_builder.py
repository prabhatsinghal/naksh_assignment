"""
profile_builder.py
Builds and manages persistent user profiles.
Computes zodiac sign from birth date and stores profile metadata.
"""

from datetime import date, datetime
from typing import Optional


# Zodiac sign date ranges
ZODIAC_RANGES = [
    ("Capricorn", (12, 22), (1, 19)),
    ("Aquarius",  (1, 20),  (2, 18)),
    ("Pisces",    (2, 19),  (3, 20)),
    ("Aries",     (3, 21),  (4, 19)),
    ("Taurus",    (4, 20),  (5, 20)),
    ("Gemini",    (5, 21),  (6, 20)),
    ("Cancer",    (6, 21),  (7, 22)),
    ("Leo",       (7, 23),  (8, 22)),
    ("Virgo",     (8, 23),  (9, 22)),
    ("Libra",     (9, 23),  (10, 22)),
    ("Scorpio",   (10, 23), (11, 21)),
    ("Sagittarius",(11, 22),(12, 21)),
    ("Capricorn", (12, 22), (12, 31)),  # Dec 22–31
]

ZODIAC_RULING_PLANETS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}


def get_zodiac_sign(birth_date: str) -> str:
    """
    Compute zodiac sign from a birth_date string in format YYYY-MM-DD.
    Returns zodiac sign name as string.
    """
    try:
        dob = datetime.strptime(birth_date, "%Y-%m-%d").date()
    except ValueError:
        return "Unknown"

    month = dob.month
    day = dob.day

    for sign, (start_m, start_d), (end_m, end_d) in ZODIAC_RANGES:
        if (month == start_m and day >= start_d) or (month == end_m and day <= end_d):
            return sign

    return "Capricorn"  # fallback for edge cases


def get_age(birth_date: str) -> Optional[int]:
    """Compute age in years from birth date string."""
    try:
        dob = datetime.strptime(birth_date, "%Y-%m-%d").date()
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except ValueError:
        return None


def build_user_profile(raw_profile: dict) -> dict:
    """
    Build an enriched user profile dict from raw API input.
    Adds computed fields: zodiac_sign, age, ruling_planet.
    """
    birth_date = raw_profile.get("birth_date", "")
    zodiac = get_zodiac_sign(birth_date)
    age = get_age(birth_date)
    ruling_planet = ZODIAC_RULING_PLANETS.get(zodiac, "Unknown")

    profile = {
        "name": raw_profile.get("name", "User"),
        "birth_date": birth_date,
        "birth_time": raw_profile.get("birth_time", ""),
        "birth_place": raw_profile.get("birth_place", ""),
        "preferred_language": raw_profile.get("preferred_language", "en"),
        "zodiac_sign": zodiac,
        "ruling_planet": ruling_planet,
        "age": age,
        # Bonus stubs — can be populated from more advanced calculations
        "moon_sign": raw_profile.get("moon_sign", None),
        "goals": raw_profile.get("goals", []),
    }
    return profile


def profile_summary(profile: dict) -> str:
    """Return a concise natural-language summary of the user profile for LLM context."""
    name = profile.get("name", "User")
    zodiac = profile.get("zodiac_sign", "Unknown")
    ruling = profile.get("ruling_planet", "Unknown")
    age = profile.get("age")
    place = profile.get("birth_place", "")
    moon = profile.get("moon_sign")
    goals = profile.get("goals", [])

    lines = [
        f"User: {name}",
        f"Zodiac Sign: {zodiac} (Ruling Planet: {ruling})",
    ]
    if age:
        lines.append(f"Age: {age} years old")
    if place:
        lines.append(f"Birth Place: {place}")
    if moon:
        lines.append(f"Moon Sign: {moon}")
    if goals:
        lines.append(f"Goals: {', '.join(goals)}")

    return "\n".join(lines)
