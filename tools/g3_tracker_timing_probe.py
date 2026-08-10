"""Where does MetricsTracker.update() spend its time, on a real mesh?

The production loop measured sec_per_step_metrics = 2.78 s out of a 3.15 s
step, and moving the accumulation on-device did not shift it. So the cost is
not the host transfer but something inside update() itself. This reproduces
the call on the same 64-device mesh with the same metric tree shape and times
each candidate separately.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app


def main(argv):
    del argv
    import time
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from utils.logging_util import MetricsTracker

    n = jax.device_count()
    print(f'devices={n} processes={jax.process_count()}')
    mesh = Mesh(np.array(jax.devices()).reshape(n), ('data',))

    # 33 leaves, matching the production metric tree, as GLOBAL SHARDED scalars
    # -- which is what p_train_step actually returns under jit/HSDP.
    def mk():
        x = jnp.arange(n, dtype=jnp.float32)
        return jax.device_put(x, NamedSharding(mesh, P('data')))
    tree = {f'm{i}': mk() for i in range(33)}
    jax.block_until_ready(list(tree.values()))

    t = MetricsTracker()
    t.update(tree); jax.block_until_ready(list(t._sum.values()))   # warm compile

    for label, fn in [
        ('update() x10', lambda: [t.update(tree) for _ in range(10)]),
        ('mean(axis=0) x33 alone', lambda: [jnp.asarray(v).mean(axis=0) for v in tree.values()]),
        ('tree add x33 alone', lambda: jax.tree.map(lambda a, b: a + b, tree, tree)),
    ]:
        t0 = time.perf_counter(); r = fn(); jax.block_until_ready(r)
        print(f'  {label:26} {time.perf_counter()-t0:7.4f} s')

    t0 = time.perf_counter(); out = t.finalize()
    print(f'  finalize() (host xfer)     {time.perf_counter()-t0:7.4f} s  leaves={len(out)}')
    return 0


if __name__ == '__main__':
    app.run(lambda argv: sys.exit(main(argv)))
