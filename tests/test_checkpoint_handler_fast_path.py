#!/usr/bin/env python3
"""Pin the two orbax knobs that make a save 5-6x faster.

WHY A TEST AND NOT JUST A COMMENT: both knobs are defended by ORBAX INTERNALS
that no public API promises.

  * `save_concurrent_bytes=None` matters only because of an `isinstance` check
    buried in `jax_array_handlers._write_arrays`: a byte limiter that is not
    `UnlimitedInFlightBytes` suppresses the OCDBT atomic transaction, so all
    1274 arrays commit their own B-tree generation (measured: 1022 commits,
    ~149 ms each, 152 s of a 236 s save). An orbax upgrade that renames that
    class, or a well-meaning edit that sets a "sensible" byte limit, silently
    restores the 152 s -- nothing errors, the checkpoint is still correct, and
    the only symptom is that the run is slow again.
  * The handler registry is copied entry-by-entry from a PRIVATE
    `_DEFAULT_TYPE_HANDLERS`. If that name moves, the obvious repair is to
    build a registry from the single jax.Array pair -- which drops the
    int/float/str handlers and kills the save on `state.step` with
    "TypeHandler lookup failed for: type=<class 'int'>".

So the test asserts against the real installed orbax, not a mock.
"""
import sys
import importlib
import types
import unittest


def _load_ckpt_util():
    # The package dir is NOT on sys.path inside a Blaze binary, so a bare
    # `import utils.ckpt_util` raises ModuleNotFoundError there while passing
    # under a plain interpreter run from the checkout root. tests/test_autoresume.py
    # solves it the same way; keep the two identical.
    import importlib
    import os
    import sys
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    return importlib.import_module('utils.ckpt_util')


class FastPytreeHandler(unittest.TestCase):

    def setUp(self):
        self.ckpt_util = _load_ckpt_util()

    def test_byte_limiter_is_unlimited_so_ocdbt_uses_a_transaction(self):
        """The whole 152 s win. `None` is the only value that yields Unlimited."""
        # WHERE THIS LIVES IS NOT STABLE. It was
        # `_src.serialization.serialization.get_byte_limiter`; a later orbax
        # moved it to `_src.serialization.limits`. Probing both is the point of
        # the test -- if NEITHER exports it, the private API this optimisation
        # rides on is gone and someone must re-verify the fast path by hand,
        # so fail loudly rather than skip.
        limits_mod = None
        for mod_path in ('orbax.checkpoint._src.serialization.limits',
                         'orbax.checkpoint._src.serialization.serialization'):
            try:
                mod = importlib.import_module(mod_path)
            except ImportError:
                continue
            if hasattr(mod, 'get_byte_limiter') and hasattr(
                    mod, 'UnlimitedInFlightBytes'):
                limits_mod = mod
                break
        self.assertIsNotNone(
            limits_mod,
            'neither orbax _src.serialization.limits nor .serialization '
            'exports get_byte_limiter/UnlimitedInFlightBytes any more. The '
            'save_concurrent_bytes=None optimisation depends on that private '
            'API; re-verify the fast path against the new orbax before '
            'trusting save timings.')
        handler = self.ckpt_util._fast_pytree_handler()
        impl = handler._handler_impl  # pylint: disable=protected-access
        limiter = limits_mod.get_byte_limiter(impl._save_concurrent_bytes)  # pylint: disable=protected-access
        self.assertIsInstance(
            limiter, limits_mod.UnlimitedInFlightBytes,
            'save_concurrent_bytes must resolve to UnlimitedInFlightBytes, or '
            'orbax skips the OCDBT transaction and every array commits its own '
            'generation (~152 s per save).')

    def test_registry_still_handles_the_scalar_types(self):
        """`state.step` is a plain int; losing its handler kills every save."""
        handler = self.ckpt_util._fast_pytree_handler()
        impl = handler._handler_impl  # pylint: disable=protected-access
        registry = impl._type_handler_registry  # pylint: disable=protected-access
        for ty in (int, float, bytes, str):
            try:
                self.assertIsNotNone(registry.get(ty))
            except Exception as exc:  # noqa: BLE001
                self.fail(f'registry lost the handler for {ty}: {exc!r}. '
                          f'Build it from a COPY of _DEFAULT_TYPE_HANDLERS, '
                          f'not from the jax.Array pair alone.')

    def test_replica_parallel_is_gated_by_size_not_disabled(self):
        """Disabling it entirely puts all 12.33 GiB on host 0 (every array is
        sharded on one axis, and replica 0's owners are devices 0-3)."""
        import jax
        handler = self.ckpt_util._fast_pytree_handler()
        impl = handler._handler_impl  # pylint: disable=protected-access
        registry = impl._type_handler_registry  # pylint: disable=protected-access
        array_handler = registry.get(jax.Array)
        self.assertIsNot(
            getattr(array_handler, '_use_replica_parallel', True), False,
            'use_replica_parallel=False concentrates the whole write on host 0')
        self.assertEqual(
            getattr(array_handler, '_min_slice_bytes_for_replica_parallel', None),
            self.ckpt_util._MIN_SLICE_BYTES_FOR_REPLICA_PARALLEL,
            'the size gate is what keeps all 8 hosts writing')

    def test_ocdbt_stays_on(self):
        """OCDBT was never the problem; the missing transaction was."""
        handler = self.ckpt_util._fast_pytree_handler()
        impl = handler._handler_impl  # pylint: disable=protected-access
        self.assertTrue(getattr(impl, '_use_ocdbt', True))


def _run_all(_argv=None):
    """Defer to absl.app.run: google3 forbids JAX calls before InitGoogle()."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(FastPytreeHandler)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    try:
        from absl import app
    except ImportError:
        _run_all()
    else:
        app.run(_run_all)
