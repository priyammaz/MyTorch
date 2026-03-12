"""
Simple tool use for calculators
"""
import signal
import threading
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
# def calculator(expression_string, timeout=5):
#     """
#     Simple calculator that takes string expressions
#     and uses sympy to evaluate them!
#     """

#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(timeout)
    
#     try:
#         expr = sympify(expression_string)
#         result = expr.evalf()
#         result = float(result)
#         ### If we can convert to int then do that ###
#         if result.is_integer():
#             result = int(result)
#         ### Otherwise round so we dont use too many tokens for numbers ###
#         else:
#             result = round(result, 4)
#         return result
#     except Exception as e:
#         print(e)
#         return None

def calculator(expression_string, timeout=5):
    """
    Simple calculator that takes string expressions
    and uses sympy to evaluate them!
    """
    result = [None]
    error = [None]

    def target():
        try:
            expr = sympify(expression_string)
            result[0] = expr.evalf()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"Calculator timed out evaluating: {expression_string}")
        return None
    if error[0] is not None:
        print(error[0])
        return None

    result = float(result[0])
    if result.is_integer():
        result = int(result)
    else:
        result = round(result, 4)
    return result


if __name__ == "__main__":
    print(calculator("4+8"))
