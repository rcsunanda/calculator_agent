from typing import Union

Number = Union[int, float]


def calculate(a: Number, b: Number, op: str) -> Number:
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0:
            raise ZeroDivisionError(f'Division by zero is not allowed. (a = {a}, b = 0)')
        return a / b
    else:
        raise ValueError(f'Unsupported operation: {op}. Supported operations are +, -, *, /.')
