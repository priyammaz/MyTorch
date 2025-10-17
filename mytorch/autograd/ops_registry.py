Ops_ = {}

def register_op(name, op_func):
    Ops_[name] = op_func

def get_op(name):
    return Ops_[name]

