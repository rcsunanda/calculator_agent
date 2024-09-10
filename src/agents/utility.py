import re
from typing import Union

Number = Union[int, float]


def validate_expression(expression: str, max_expression_length: int) -> bool:
    if len(expression) > max_expression_length:
        raise ValueError(f"Expression exceeds maximum length of {max_expression_length} characters")

    pattern = r'^[\d\s\+\-\*\/\(\)\.]+$'  # Only digits, spaces, and basic arithmetic operators are allowed

    if re.match(pattern, expression) is None:
        raise ValueError(f"Invalid characters in the expression: {expression}")

    return True


def float_to_str(f: float) -> str:
    return f'{f: g}'


def create_number_pattern(num: Union[int, float]) -> str:
    if isinstance(num, int):
        return r'\b' + str(num) + r'\b'
    else:
        # Handle both integer and decimal representations
        return r'\b' + float_to_str(num).replace('.', r'\.?') + r'\b'


def reduce_expression(expression: str, a: Number, b: Number, op: str, result: Number) -> str:
    pattern = create_number_pattern(a) + r'\s*' + re.escape(op) + r'\s*' + create_number_pattern(b)
    new_expression = re.sub(pattern, float_to_str(result), expression, count=1)
    return new_expression
