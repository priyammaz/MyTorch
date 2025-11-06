"""
Im not a huge fan of this code, but its for learning so its ok! 

KEY IDEA: Training models requires the storage of intermediate variable for backprop. All of our
          modules until now have been capturing these variables as closures. But this increases the 
          memory cost for training! So what if we did the forward pass like normal, but without 
          any gradients. And then in the backward pass we REDO the forward pass again to get those
          intermediate variables.


WHY THIS CODE SUCKS: The problem is there is a memory leak somewhere. Somehow between the way I defined
my computational graph with closures and how I am doing the recomputation of a subgraph here (when applied to 
complex ops like an entire transformer block) there is some cycle references that the garbage collector doesn't
clear out. 

So what we do is add an extra Global variable _checkpoint_backward_count, and every iter_gc backward passes we 
trigger a manual Garbage Collection. Calling GC every backward pass is prohibitively slow, so this is a reasonable
balance to call it every 10 iterations (the default)

GAINS: We get to train models with a 20-30% decrease in memory use but as expected training slows down due to the recompute
of the forward pass (i measure about 20-40% slowdown in training speed of GPT2 when wrapping the entire transformer block in 
checkpointing)

"""
from .. import Tensor, no_grad
import gc

_checkpoint_backward_count = 0

def checkpoint(fn, *args, iter_gc=20):
    """
    When checkpointing we do not store any of the intermediates
    in the forward pass (no_grad basically) and call forward again
    in the backward pass to recompute everything
    """

    ### Get the name and __call__ of the function if it exists ###
    ### as we could have nn.Modules or just functions ###
    name = getattr(fn, "__name__", fn.__class__.__name__)
    fn = getattr(fn, "__call__", fn)

    ### No need for grads in the forward pass, we can detach everything ###
    detached_args = []
    for arg in args:
        if isinstance(arg, Tensor):
            detached_args.append(arg.detach())
        else:
            detached_args.append(arg)
    
    ### Forward without Grads ###
    with no_grad():
        out = fn(*detached_args)
    
    def backward_fn(grad_output):

        ### Create detached args but with requires_grad=True to act as leaf nodes in a new computational graph
        ### this basically says to treat each input as an input parameter to some 
        ### module that needs grads. This way
        recompute_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                r_arg = arg.detach()
                r_arg.requires_grad = True
                recompute_args.append(r_arg)
            else:
                recompute_args.append(arg)

        ### Recompute the forward pass and grads in this new subgraph ###
        recomputed_output = fn(*recompute_args)
        recomputed_output.backward(grad_output)
        
        ### Transfer gradients from recompute leaves to original args ###
        ### as they will hold onto this grad for whatever backprop comes before them ###
        for orig_arg, recomp_arg in zip(args, recompute_args):
            if isinstance(orig_arg, Tensor):
                if orig_arg.grad is None:
                    orig_arg.grad = recomp_arg.grad
                else:
                    orig_arg.grad += recomp_arg.grad

        # Explicit cleanup to help GC
        for r in recompute_args:
            if isinstance(r, Tensor):
                r.grad = None
                r.grad_fn = None
                r._parents = None
        
        # Clear recomputed_output references
        if isinstance(recomputed_output, Tensor):
            recomputed_output.grad = None
            recomputed_output.grad_fn = None
            recomputed_output._parents = None

        ### Clear the entire output ###
        recomputed_output = None

        ### I DONT LIKE THIS CODE BUT ITS FINE I GUESS ###
        global _checkpoint_backward_count
        _checkpoint_backward_count += 1
        if _checkpoint_backward_count % iter_gc == 0:
            gc.collect()

    out = Tensor(
        out.data if isinstance(out, Tensor) else out,
        requires_grad=True,
        grad_fn=backward_fn,
        grad_fn_name=f"<Checkpoint{name}Backward>"
    )

    out._add_parents(*args)
    
    return out