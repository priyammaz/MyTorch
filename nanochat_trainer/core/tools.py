"""
Simple tool use for calculators
"""
import signal
from sympy import sympify

### TOOLUSE TIMOUT HANDLER ###
class Timeout(Exception):
    """Wrapper timout exception"""
    pass

def handler(*args, **kwargs):
    """
    When the OS delivers a sigalrm signal, the handler
    is called, which then raises a Timeout
    """
    raise Timeout("Calculator Timeout!!")

### CALCULATOR TOOL ###
def calculator(expression_string, timeout=5):
    """
    Simple calculator that takes string expressions
    and uses sympy to evaluate them!
    """

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        expr = sympify(expression_string)
        result = expr.evalf()
        result = float(result)
        ### If we can convert to int then do that ###
        if result.is_integer():
            result = int(result)
        ### Otherwise round so we dont use too many tokens for numbers ###
        else:
            result = round(result, 4)
        return result
    except Exception as e:
        print(e)
        return None
