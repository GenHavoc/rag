import re

SECTION_PATTERNS = {
    "Abstract": r"^abstract$",
    "Introduction": r"^introduction$",
    "Methods": r"^(methods?|methodology|materials and methods)$",
    "Experiments": r"^(experiment|experimental setup)$",
    "Results": r"^(results?|evaluation|findings)$",
    "Discussion": r"^discussion$",
    "Conclusion": r"^(conclusion|future work)$"
}

def detect_section(text):
    lines = text.lower().splitlines()

    for line in lines[:10]:
        line = line.strip()
        for section, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, line):
                return section

    return "Body"
