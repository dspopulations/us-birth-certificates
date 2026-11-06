"""Parse explicit year selections for import and optional export commands."""

import argparse
import re


def parse_years(value: str) -> list[int]:
    years = set()
    for part in value.split(","):
        match = re.fullmatch(r"\s*(\d{4})(?:-(\d{4}))?\s*", part)
        if not match:
            raise argparse.ArgumentTypeError(
                "Use years such as 1989-2025 or 1989,2003,2025"
            )
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1989 <= first <= last <= 2025:
            raise argparse.ArgumentTypeError(
                "Years must be in ascending ranges within 1989-2025"
            )
        years.update(range(first, last + 1))
    return sorted(years)
