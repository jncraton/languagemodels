from sympy import symbols, Eq
from sympy import diff, integrate, simplify, Matrix, limit, oo
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve

def process_diff(input_str):
    parts = input_str.split(' ')
    if len(parts) != 3:
        raise ValueError(f"Invalid number of arguments for diff: {input_str}")
    _, expr, var = parts
    derivative = diff(parse_expr(expr), symbols(var))
    return f'The derivative of {expr} with respect to {var} is {derivative}'

def process_integrate(input_str):
    parts = input_str.split(' ')
    if len(parts) != 3:
        raise ValueError(f"Invalid number of arguments for integrate: {input_str}")
    _, expr, var = parts
    integral = integrate(parse_expr(expr), symbols(var))
    return f'The integral of {expr} with respect to {var} is {integral}'

def process_limit(input_str):
    parts = input_str.split(' ')
    if len(parts) != 4:
        raise ValueError(f"Invalid number of arguments for limit: {input_str}")
    _, expr, var, direction = parts
    if direction == '+':
        limit_solution = limit(parse_expr(expr), symbols(var), oo)
    elif direction == '-':
        limit_solution = limit(parse_expr(expr), symbols(var), -oo)
    else:
        raise ValueError(f'Invalid direction: {direction}')
    return f'The limit of {expr} as {var} approaches {direction}∞ is {limit_solution}'

def process_eigenvalues(input_str):
    _, matrix_str = input_str.split(' ', 1)
    matrix = Matrix(eval(matrix_str))
    eigenvalues = matrix.eigenvals()
    return f'The eigenvalues of the matrix {matrix} are {eigenvalues}'

def process_simplify(input_str):
    _, expr = input_str.split(' ', 1)
    simplification = simplify(parse_expr(expr))
    return f'The simplification of {expr} is {simplification}'

def process_system(input_str):
    equations = input_str.split(';')
    symbols_set = set()
    eqs = []
    for eq in equations:
        eq = eq.strip()
        left, right = eq.split('=')
        symbols_in_eq = [symb for symb in parse_expr(eq).free_symbols]
        symbols_set.update(symbols_in_eq)
        eqs.append(Eq(parse_expr(left), parse_expr(right)))
    solutions = solve(eqs, list(symbols_set))
    return ', '.join([f'{str(k)} is {v}' for k, v in solutions.items()])

def resolve(input_str):
    try:
        input_list = input_str.split(' ', 1)
        op = input_list[0]
        expr = input_list[1] if len(input_list) > 1 else None

        ops = {
            'diff': process_diff,
            'integrate': process_integrate,
            'limit': process_limit,
            'eigenvalues': process_eigenvalues,
            'simplify': process_simplify,
        }

        if op in ops and expr is not None:
            return ops[op](input_str)
        else:
            if ';' in input_str:
                return process_system(input_str)
            elif '=' in input_str:
                equation = Eq(*map(parse_expr, input_str.split('=')))
                solutions = solve(equation)
                if isinstance(solutions, dict):
                    return ', '.join([f'{str(k)} is {v}' for k, v in solutions.items()])
                elif isinstance(solutions, list):
                    return ', '.join([f'x is {sol}' for sol in solutions])
            else:
                return parse_expr(input_str)
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

"""
# Examples
print(resolve("diff x**2 x"))  # The derivative of x**2 with respect to x is 2*x
print(resolve("integrate x**2 x"))  # The integral of x**2 with respect to x is x**3/3
print(resolve("limit x**2/x x +"))  # The limit of x**2/x as x approaches +∞ is ∞
print(resolve("eigenvalues [[1, 2], [2, 1]]"))  # The eigenvalues of the matrix Matrix([[1, 2], [2, 1]]) are {3: 1, -1: 1}
print(resolve("2*x=6"))  # x is 3
print(resolve("simplify 2*2 + 3*3"))  # Returns 'The simplification of 2*2 + 3*3 is 13'
"""