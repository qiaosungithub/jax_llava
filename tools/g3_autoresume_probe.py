"""Exercise the auto-resume probe inside a real Blaze binary, against real CNS.

This is the local check that the workstation harness could NOT make: there,
`ckpt_util.FS` was a fileutil stand-in. Here it is the actual `_GfileFS` that
will run on Borg, reading the actual checkpoints the smoke runs wrote.
"""
import os
import sys

# Same shape as tools/g3_config_probe.py: this file lives in tools/, but
# jax_llava's modules are imported as top-level names (`utils`, `train`), so
# the package root has to be on sys.path before any of them resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app  # noqa: E402


def main(argv):
    del argv
    from utils import ckpt_util

    print(f'in_google3        = {ckpt_util.g3_env.in_google3()}')
    print(f'FS                = {type(ckpt_util.FS).__name__}')

    buckets = {
        '277049823': '/cns/yucmhcg-d/home/qiaos/jax_llava/logs/jax_llava/'
                     'xid_277049823_20260804_180552_jax-llava-smoke14-metrics',
        '277043477': '/cns/yucmhcg-d/home/qiaos/jax_llava/logs/jax_llava/'
                     'xid_277043477_20260804_173533_jax-llava-smoke13',
    }

    class Cfg(dict):
        pass

    ok = True
    for xid, bucket in buckets.items():
        print(f'\n=== XID {xid} ===')
        root = f'{bucket}/checkpoints'
        names = sorted(ckpt_util._listdir(root))
        print(f'  entries: {names}')
        for name in names:
            print(f'    {name:30s} -> {ckpt_util._checkpoint_is_complete(f"{root}/{name}")}')
        latest = ckpt_util.latest_complete_checkpoint(root)
        print(f'  latest_complete_checkpoint = {latest}')
        os.environ['CHECKPOINT_BUCKET'] = bucket
        got, why = ckpt_util.resolve_borg_autoresume(Cfg())
        print(f'  resolve_borg_autoresume    = {got!r} why_not={why!r}')
        if got != f'{root}/checkpoint_12':
            print(f'  *** UNEXPECTED: wanted {root}/checkpoint_12')
            ok = False

        # checkpoint_step() is what train.py calls on the resolved path; if it
        # cannot parse the step, the resume dies inside _train_llava_curriculum.
        step = ckpt_util.checkpoint_step(got, zone='us-east5')
        print(f'  checkpoint_step()          = {step}')
        if step != 12:
            print('  *** UNEXPECTED step'); ok = False

        # The explicit-request branch must still win in the real binary.
        got2, why2 = ckpt_util.resolve_borg_autoresume(Cfg(load_from='/cns/x/checkpoint_3'))
        print(f'  with explicit load_from    = {got2!r} why_not={why2!r}')
        if got2 is not None:
            print('  *** UNEXPECTED: explicit load_from did not win'); ok = False

    # The stage-boundary path resolves through the DURABLE pretrained prefix.
    print('\n=== convert_to_pretrained_gs (stage-boundary durable path) ===')
    for xid, bucket in buckets.items():
        os.environ['CHECKPOINT_BUCKET'] = bucket
        src = f'{bucket}/checkpoints'
        dst = ckpt_util.convert_to_pretrained_gs(src, zone='us-east5')
        src_cell = ckpt_util._cns_cell(src)
        dst_cell = ckpt_util._cns_cell(dst)
        print(f'  {xid}: {src}\n      -> {dst}')
        print(f'      cell {src_cell} -> {dst_cell}  same_cell={src_cell == dst_cell}')
        if src_cell != dst_cell:
            print('  *** UNEXPECTED: durable path crossed cells'); ok = False
        print(f'      pretrained dir entries: {sorted(ckpt_util._listdir(dst))}')

    print('\nRESULT:', 'ALL EXPECTED' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    app.run(main)
