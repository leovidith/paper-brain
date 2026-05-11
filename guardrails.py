import re

BLOCKED_PHRASES = {
    "ignore previous instructions", "jailbreak", "bypass your rules",
    "forget your prompt", "act as", "you are now", "dan mode"
}

AADHAAR_KEYWORDS = {"aadhaar", "aadhar", "uidai", "uid"}
PHONE_KEYWORDS = {"phone", "mobile", "contact", "call", "number"}

EMAIL_REGEX    = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}')
PHONE_REGEX    = re.compile(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)(\d{3,4}[\s\-]?\d{3,4})')
SSN_REGEX      = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CARD_REGEX     = re.compile(r'\b(?:\d[ -]?){13,16}\b')
AADHAAR_REGEX  = re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b')
PAN_REGEX      = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
PASSPORT_REGEX = re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b')
IP_REGEX       = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]

VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]
]

REDACTION_MAP = {
    "email": "[REDACTED_EMAIL]", "SSN": "[REDACTED_SSN]",
    "PAN": "[REDACTED_PAN]", "passport": "[REDACTED_PASSPORT]",
    "IP address": "[REDACTED_IP]", "credit card": "[REDACTED_CARD]",
    "Aadhaar": "[REDACTED_AADHAAR_NUMBER]", "phone": "[REDACTED_PHONE_NUMBER]"
}

HALLUCINATION_SIGNALS = [
    "as an ai language model", "i cannot browse the internet",
    "my knowledge cutoff", "i don't have access to real-time information",
    "i was not trained on",
]


def context_exists(text: str, start: int, end: int, keywords: set, window: int = 30) -> bool:
    left = max(0, start - window)
    right = min(len(text), end + window)
    context = text[left:right].lower()
    return any(keyword in context for keyword in keywords)


def verhoeff_validate(number: str) -> bool:
    c = 0
    for i, digit in enumerate(map(int, reversed(number))):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]
    return c == 0


def luhn_validate(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def valid_ip(ip: str) -> bool:
    try:
        return all(0 <= int(part) <= 255 for part in ip.split('.'))
    except:
        return False


def redact_simple_patterns(text: str) -> str:
    text = EMAIL_REGEX.sub(REDACTION_MAP["email"], text)
    text = SSN_REGEX.sub(REDACTION_MAP["SSN"], text)
    text = PAN_REGEX.sub(REDACTION_MAP["PAN"], text)
    text = PASSPORT_REGEX.sub(REDACTION_MAP["passport"], text)
    return text


def redact_ip_addresses(text: str) -> str:
    matches = list(IP_REGEX.finditer(text))
    for match in reversed(matches):
        if valid_ip(match.group()):
            text = text[:match.start()] + REDACTION_MAP["IP address"] + text[match.end():]
    return text


def redact_credit_cards(text: str) -> str:
    matches = list(CARD_REGEX.finditer(text))
    for match in reversed(matches):
        digits = re.sub(r'\D', '', match.group())
        if 13 <= len(digits) <= 16 and luhn_validate(digits):
            text = text[:match.start()] + REDACTION_MAP["credit card"] + text[match.end():]
    return text


def redact_aadhaar(text: str) -> str:
    matches = list(AADHAAR_REGEX.finditer(text))
    for match in reversed(matches):
        digits = re.sub(r'\D', '', match.group())
        if len(digits) != 12:
            continue
        score = 0.4
        if context_exists(text, match.start(), match.end(), AADHAAR_KEYWORDS):
            score += 0.3
        if verhoeff_validate(digits):
            score += 0.5
        if score >= 0.7:
            text = text[:match.start()] + REDACTION_MAP["Aadhaar"] + text[match.end():]
    return text


def redact_phone_numbers(text: str) -> str:
    matches = list(PHONE_REGEX.finditer(text))
    for match in reversed(matches):
        digits = re.sub(r'\D', '', match.group())
        if 10 <= len(digits) <= 13:
            if context_exists(text, match.start(), match.end(), PHONE_KEYWORDS):
                text = text[:match.start()] + REDACTION_MAP["phone"] + text[match.end():]
    return text


def sanitize_query(query: str) -> str:
    query = redact_simple_patterns(query)
    query = redact_ip_addresses(query)
    query = redact_credit_cards(query)
    query = redact_aadhaar(query)
    query = redact_phone_numbers(query)
    return query


def input_guardrail(query: str) -> tuple[bool, str]:
    lower = query.lower()
    for phrase in BLOCKED_PHRASES:
        if phrase in lower:
            return False, f"Query blocked: contains disallowed phrase '{phrase}'."
    sanitized = sanitize_query(query)
    return True, sanitized


def output_guardrail(answer: str, source_docs: list) -> tuple[str, bool]:
    a_lower = answer.lower()
    for signal in HALLUCINATION_SIGNALS:
        if signal in a_lower:
            warning = "\n\n⚠️ [Output Warning: Response may reference information outside the provided documents.]"
            return answer + warning, True
    if not source_docs:
        if "i don't know" not in a_lower and "no information" not in a_lower:
            warning = "\n\n⚠️ [Output Warning: No source documents were retrieved. This answer may be hallucinated.]"
            return answer + warning, True
    return answer, False