"""
Triton default autotuner uses Torch, this is a replica of their 
autotuner using cupy instead!
"""
import cupy as cp

def _summarize_statistics(values, quantiles=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median", "all"]

    values = sorted(values)

    if quantiles is not None:
        def percentile(q):
            # Linear interpolation between neighboring samples.
            pos = q * (len(values) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(values) - 1)
            frac = pos - lo
            return values[lo] * (1 - frac) + values[hi] * frac

        return [percentile(q) for q in quantiles]

    if return_mode == "min":
        return values[0]
    if return_mode == "max":
        return values[-1]
    if return_mode == "mean":
        return sum(values) / len(values)
    if return_mode == "median":
        n = len(values)
        mid = n // 2
        if n % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2
    
    if return_mode == "all":
        return values

def do_bench_cudagraph(
    fn,
    rep=20,
    grad_to_none=None,
    quantiles=None,
    return_mode="mean",
):

    assert return_mode in ["min", "max", "mean", "median", "all"]

    stream = cp.cuda.Stream(non_blocking=True)

    with stream:
        # Warmup
        fn()
        stream.synchronize()

        # Reset gradients before graph capture.
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record(stream)

        for _ in range(5):
            fn()

        end_event.record(stream)
        end_event.synchronize()

        estimate_ms = cp.cuda.get_elapsed_time(
            start_event,
            end_event,
        ) / 5

        if estimate_ms == 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, int(rep / estimate_ms))

        stream.begin_capture()

        for _ in range(n_repeat):
            fn()

        graph = stream.end_capture()

        ret = []
        n_retries = 10

        for _ in range(n_retries):

            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record(stream)
            graph.launch(stream)
            end_event.record(stream)
            end_event.synchronize()

            elapsed_ms = cp.cuda.get_elapsed_time(
                start_event,
                end_event,
            )

            ret.append(elapsed_ms / n_repeat)

        return _summarize_statistics(
            ret,
            quantiles=quantiles,
            return_mode=return_mode,
        )