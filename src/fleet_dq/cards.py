"""Fuel card numbers: 18 digits ending in a Luhn check digit, and the damage Excel does to them."""

ISSUER = "707781"
EXCEL_DIGITS = 15


def luhn_check_digit(body: str) -> str:
    """The digit that makes `body` plus that digit pass the Luhn check."""
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def luhn_valid(number: str) -> bool:
    return len(number) > 1 and number.isdigit() and luhn_check_digit(number[:-1]) == number[-1]


def card_number(account: int, sequence: int) -> str:
    """Issuer (6 digits), operator account (6), card sequence (5), check digit (1)."""
    body = f"{ISSUER}{account:06d}{sequence:05d}"
    return body + luhn_check_digit(body)


def excel_rounded(number: str) -> str:
    """What Excel stores for a long number typed or pasted as a number: digits after the 15th become zero."""
    return number[:EXCEL_DIGITS] + "0" * (len(number) - EXCEL_DIGITS)
