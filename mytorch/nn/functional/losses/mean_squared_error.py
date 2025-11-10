from mytorch import Tensor
from mytorch.nn.functional.utils import get_inner_array

def auto_mse(pred, labels):
    return ((pred - labels)**2).mean()

def manual_mse(pred, labels):
    diff = get_inner_array(pred) - get_inner_array(labels)
    out_data = (diff**2).mean()

    def _mse_backward(grad_output):

        N = diff.shape[0]
        grad_input = (2.0 / N) * diff * grad_output

        if pred.grad is None:
            pred.grad = grad_input
        else:
            pred.grad += grad_input
    
    requires_grad = pred.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_mse_backward if requires_grad else None,
        grad_fn_name="<MSEBackward>" if requires_grad else None
    )

    if requires_grad:
        out._add_parents(pred)
    
    return out

def mse_loss(pred, labels, auto=False):
    if auto:
        return auto_mse(pred, labels)
    else:
        return manual_mse(pred, labels)
