# jax_llava as a google3 Bazel binary, for Borg/TPU via XManager.
#
# The GCP TPU-VM path (`python main.py ...` on a kmh-nfs mount) still works
# unchanged; everything google3-specific in the sources is behind
# `utils/g3_env.py::in_google3()`, which is an import check for
# `google3.pyglib`, i.e. true exactly when this BUILD produced the binary.
#
# `py_binary`, NOT `pytype_binary`. The pytype aspect runs pyrefly, which
# hard-fails on `import wandb`, `from pyglib import gfile` and
# `torch.multiprocessing.get_context` even with `strict_deps = False` and
# `tags = ["ignore_pytype"]`. Plain py_binary skips the aspect entirely.

load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

py_binary(
    name = "main",
    srcs = ["main.py"],
    data = glob(
        [
            "**/*.py",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
        ],
        exclude = ["main.py"],
    ),
    main = "main.py",
    # The rest of jax_llava (train.py, input_pipeline.py, utils/, models/,
    # evals/) travels in `data` and is imported by path, not as Bazel targets.
    # Strict deps therefore cannot see `import train` and rejects the build; it
    # is not wrong, it just does not model this layout. The launcher already
    # passes --norun_validations for the same reason
    # (tpu_cmd/xm_launcher.py: bazel_args); `nostrictdeps` is the in-BUILD
    # equivalent, so a plain `blaze build` behaves like a launched one -- which
    # matters, because the local build is what catches import errors before
    # they cost a Borg round trip.
    #
    # Making these real py_library targets is the eventual fix, but it means
    # untangling a 26k-line codebase's import graph, and every module that is
    # missed becomes a runtime failure on the remote machine instead of a
    # build failure here. Not a good trade while the port is being proven.
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [
        # ---- the in-tree webdataset replacement -------------------------
        # There is no //third_party/py/webdataset in google3 and nothing in
        # the depot links the real package. wds_shim reimplements the ~7
        # symbols jax_llava uses and, unlike a pip webdataset, reads shards
        # straight from CNS. main.py registers it as sys.modules["webdataset"].
        ":wds_shim",

        # ---- filesystems -------------------------------------------------
        # Everything durable (data, checkpoints, mirrored logs) is on CNS. A
        # Borg task runs as <user>@prod.google.com, which cannot write our
        # gs:// buckets, so this is not optional.
        "//file/colossus/public:cns",
        "//pyglib:gfile",
        # Registers the /tfhub/ prefix with the File API. gemma's checkpoints
        # live at /tfhub/prod/g_mini/GEMMA-3.0-1B-IT-ORBAX/1, and without this
        # the open fails on an unknown prefix rather than on permissions.
        "//file/tfhub",  # buildcleaner: keep

        # ---- multiprocessing ---------------------------------------------
        # sys.executable is None inside a Blaze py_binary, so stdlib spawn
        # cannot re-exec us and DataLoader workers never start.
        # g3_multiprocessing.handle_main teaches it how (main.py calls it).
        # `fork` is not an alternative: torch's google3 multiprocessing
        # asserts on it (go/python-tips/018) and forking after JAX has
        # started deadlocks.
        "//pyglib/contrib/g3_multiprocessing",

        # ---- JAX ----------------------------------------------------------
        "//third_party/py/jax",
        # Registers the TPU PJRT backend. `//third_party/py/jax` ALONE gives a
        # CPU-ONLY binary: jax_google deliberately does not depend on the TPU
        # compiler, and its `_tpu_backend_factory` is registered with
        # `fail_quietly=True`, so the resulting "No TPU backend found"
        # RuntimeError is swallowed into a logger.info and JAX falls back to
        # CPU with no warning at INFO or above. A sibling project lost 2.5h on
        # a v6p-16 at a 0.000 duty cycle to exactly this; main.py's
        # _assert_accelerator_backend() is the belt to this braces.
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/flax",
        "//third_party/py/optax",
        "//third_party/py/orbax/checkpoint",
        "//third_party/py/chex",

        # ---- data pipeline -------------------------------------------------
        # :pytorch is the py_library; the bare //third_party/py/torch is a
        # cc_library and is the wrong label for a py_binary.
        "//third_party/py/torch:pytorch",
        "//third_party/py/torchvision",
        # StatefulDataLoader, for exact loader resume. Its visibility names
        # torchtitan only, but //experimental/ builds against it.
        "//third_party/py/torchdata",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/pandas",
        # Only reached for gs:// roots (the GCP path); no gcsfs in google3, so
        # a gs:// glob would fail -- which is correct, since a Borg job must
        # read CNS.
        "//third_party/py/fsspec",

        # ---- model ---------------------------------------------------------
        # Must resolve to v4 (4.57.6). `:v5` deleted all Flax modelling and
        # jax_llava needs FlaxCLIPVisionModel.
        "//third_party/py/transformers",
        "//third_party/py/gemma/gm",
        "//third_party/py/sentencepiece",
        # google3 spells the SP proto module
        # google3.third_party.sentencepiece.src.sentencepiece_model_pb2, NOT
        # `from sentencepiece import sentencepiece_model_pb2`.
        "//third_party/sentencepiece/src:sentencepiece_model_py_pb2",
        "//third_party/py/huggingface_hub",

        # ---- config / misc --------------------------------------------------
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/ml_collections",
        "//third_party/py/ml_collections/config_flags",
        "//third_party/py/yaml",
        "//third_party/py/etils/epath",
        # A no-op mock (init/log/finish/Table/plot/Video/run only, and it
        # stores nothing). jax_llava wraps every call in try/except and
        # use_wandb defaults False, so it degrades safely -- but do not build
        # a metrics story on it.
        "//third_party/py/scamper:wandb_mock",
    ],
)

py_library(
    name = "wds_shim",
    srcs = ["wds_shim/__init__.py"],
    deps = [
        # Reads shards from /cns/, /bigstore/, /placer/ -- the thing a
        # pip-installed webdataset fundamentally cannot do.
        "//pyglib:gfile",
        "//third_party/py/PIL:pil",
    ],
)
