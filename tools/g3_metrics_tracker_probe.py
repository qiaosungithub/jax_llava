"""The rewritten MetricsTracker must produce the SAME numbers, only faster.

A performance fix that changes a reported metric is a worse bug than the one
it fixes, so this compares the new device-side accumulation against the old
host-side one on identical inputs, including the per-replica (leading device
axis) shape that the training loop actually produces.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app


def main(argv):
    del argv
    # google3 refuses JAX before absl.app.run(); import inside main().
    import jax
    import jax.numpy as jnp
    import numpy as np
    from utils.logging_util import MetricsTracker

    class OldTracker:
        def __init__(self): self._sum=None; self._n=0
        @staticmethod
        def _m(x):
            a=np.asarray(jax.device_get(x))
            if a.ndim>=1: a=a.mean(axis=0)
            return a
        def update(self,t):
            lm=jax.tree.map(self._m,t)
            self._sum=lm if self._sum is None else jax.tree.map(lambda s,x:s+x,self._sum,lm)
            self._n+=1
        def finalize(self):
            if not self._n: return {}
            out=jax.tree.map(lambda s: float(np.asarray(s/self._n,dtype=np.float64).mean()), self._sum)
            self._sum,self._n=None,0
            return out

    rng=np.random.default_rng(0)
    nd=jax.local_device_count()
    for shape,label in [((nd,),'per-replica'), ((),'scalar'), ((nd,4),'per-replica vector')]:
        new,old=MetricsTracker(),OldTracker()
        for _ in range(7):
            tree={'loss':jnp.asarray(rng.normal(size=shape)),
                  'acc': jnp.asarray(rng.random(size=shape)),
                  'grad_norm': jnp.asarray(rng.random(size=shape)*100)}
            new.update(tree); old.update(tree)
        a,b=new.finalize(),old.finalize()
        ok=all(abs(a[k]-b[k])<1e-9 for k in a)
        print(f"{'PASS' if ok else 'FAIL'}  {label:22} new={ {k:round(v,6) for k,v in a.items()} }")
        if not ok:
            print("       old=",{k:round(v,6) for k,v in b.items()}); return 1
    print("\nvalues identical; empty tracker:", MetricsTracker().finalize())
    return 0


if __name__ == '__main__':
    app.run(lambda argv: sys.exit(main(argv)))
