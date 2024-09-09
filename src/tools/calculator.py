def calculate(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0:
            raise ZeroDivisionError('Division by zero is not allowed (b = 0)')
        return a / b
    else:
        raise ValueError(f'Invalid operation: {op}. Supported operations are +, -, *, /.')
