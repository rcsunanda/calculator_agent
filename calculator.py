def calculate(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0:
            raise ZeroDivisionError('b = 0')
        return a / b
    else:
        raise ValueError(f'Invalid op. {op}')
