import weakref

class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.needs_input_grad = None

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

class Function:
    @classmethod
    def apply(cls, *inputs):

        ### Lazy Import ###
        from ..tensor import Tensor

        ### Create Context Object ##
        ctx = Context()

        # Track which inputs require grad
        ctx.needs_input_grad = tuple(
            getattr(t, "requires_grad", False) if isinstance(t, Tensor) else False
            for t in inputs
        )

        # Extract raw data
        raw_inputs = [t.data if isinstance(t, Tensor) else t for t in inputs]
        
        # Run forward pass
        outputs = cls.forward(ctx, *raw_inputs)

        requires_grad = any(ctx.needs_input_grad) and Tensor.build_graph_enabled()

        outputs = Tensor(outputs,
                         requires_grad=requires_grad,
                         grad_fn=(lambda g: cls.backward(ctx, g)),
                         grad_fn_name=f"<{cls.__name__}Backward>")
        
        if requires_grad:
            # Store parents
            parents = [weakref.ref(arg) for arg in inputs if isinstance(arg, Tensor)]
            outputs._add_parents(*parents)

        return outputs

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
    


