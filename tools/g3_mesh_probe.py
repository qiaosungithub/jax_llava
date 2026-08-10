"""What mesh does this slice actually get, and does the topology table know it?

`get_mesh()` looks up `device_kind` in TOPOLOGIES and, on no match, SILENTLY
falls back to a flat 1-D mesh meant for CPU/GPU debugging. A 1-D mesh is not a
crash -- training runs, and only the throughput says anything is wrong -- so
the fallback has to be probed deliberately.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app


def main(argv):
    del argv
    import jax
    from utils import pjit_util

    kind = jax.local_devices()[0].device_kind
    print(f'device_kind : {kind!r}')
    print(f'devices     : {jax.device_count()} over {jax.process_count()} processes')
    matched = [k for k in pjit_util.TOPOLOGIES if k in kind.lower()]
    print(f'TOPOLOGIES keys: {sorted(pjit_util.TOPOLOGIES)}')
    print(f'matched key : {matched or "NONE -> 1-D debug fallback"}')

    mesh = pjit_util.get_mesh()
    print(f'mesh shape  : {mesh.devices.shape}  axes={mesh.axis_names}')
    if len(mesh.devices.shape) == 1:
        print('VERDICT: FLAT 1-D MESH. Under hsdp/hsdp_legacy_data the model axis is')
        print('         axis_names[-1] -- the only axis -- so parameters are sharded')
        print('         across ALL devices and every matmul pays a full-mesh collective.')
        return 1
    print('VERDICT: multi-axis mesh, data/model axes are distinct.')
    return 0


if __name__ == '__main__':
    app.run(lambda argv: sys.exit(main(argv)))
