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
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

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
        # The metric sink that actually stores anything. google3's wandb is a
        # no-op mock, so utils/g3_metrics.py mirrors every write_scalars() into
        # DeepMind Datatables through CLU, which backs
        # http://datatable/xid/<XID>/data and http://flatboard/xid/<XID>.
        # The `:notf` variant avoids pulling in TensorFlow *for CLU*; note the
        # binary still contains TF, because flax.io runs on tensorflow.io.gfile
        # and that is precisely why flax's latest_checkpoint reaches /cns/.
        # CLU is //visibility:public, unlike the datatables client itself.
        "//third_party/py/clu/metric_writers:notf",
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

# Step 2 of the port: pull REAL batches from the REAL CNS shards, locally,
# before paying for a Borg round trip. Shares main's dep list because it
# imports main (which installs the webdataset shim) and then jax_llava's own
# create_split -- what it proves is what training will do, not a lookalike.
py_binary(
    name = "g3_dataloader_probe",
    srcs = ["tools/g3_dataloader_probe.py"],
    data = glob(
        [
            "**/*.py",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
        ],
        exclude = ["tools/g3_dataloader_probe.py"],
    ),
    main = "tools/g3_dataloader_probe.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [":main"],
)

# Dry-runs everything that happens BEFORE the first training step: config
# merge, curriculum stage expansion, dataset-root resolution, the locality
# guard, and where the weights and checkpoints would come from and go to.
# Seconds, versus finding a bad config after packaging and scheduling.
py_binary(
    name = "g3_config_probe",
    srcs = ["tools/g3_config_probe.py"],
    data = glob(
        [
            "**/*.py",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
        ],
        exclude = ["tools/g3_config_probe.py"],
    ),
    main = "tools/g3_config_probe.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [":main"],
)

# Auto-resume, proven against the REAL checkpoints before a launch.
#
# The workstation can only test the decision logic over a stubbed filesystem:
# outside a Blaze binary `ckpt_util.FS` is not the gfile-backed one that runs
# on Borg, and /cns/ is unreachable in-process. This target closes that gap --
# same binary shape, same `_GfileFS`, same real checkpoint bytes -- and it
# answers in seconds instead of a package + schedule + 5 min of imports.
py_binary(
    name = "g3_autoresume_probe",
    srcs = ["tools/g3_autoresume_probe.py"],
    data = glob(
        [
            "**/*.py",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
        ],
        exclude = ["tools/g3_autoresume_probe.py"],
    ),
    main = "tools/g3_autoresume_probe.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [":main"],
)

# The CNS log mirror, tested directly and in SECONDS.
#
# On this workstation the mirror is the only readable log a Borg task leaves
# (`borg tasklog` SIGABRTs on PERMISSION_DENIED, `analog --remote` is refused,
# Coroner's binary is unreadable, and the task is GC'd from the borgmaster in
# minutes). A bug in it does not lose one log, it makes every LATER failure
# undiagnosable -- so it gets its own proof rather than being inferred from a
# training run that must also survive TPU bring-up and 5 min of imports.
#
# Deliberately does NOT depend on `:main`: the point is a build-and-run cycle
# measured in seconds, and the mirror needs pyglib.gfile, etils and absl and
# nothing else. Pulling torch in would make this as slow as the thing it is
# meant to de-risk.
py_binary(
    name = "g3_mirror_probe",
    srcs = ["tools/g3_mirror_probe.py"],
    # `data`, not `srcs`, and the same glob shape the other targets use. The
    # rest of this project is imported BY PATH (`from utils import ...`, with
    # the package dir on sys.path) rather than as Bazel targets, so a module
    # listed in `srcs` lands at its google3-relative path and `import utils`
    # then fails with ModuleNotFoundError at runtime despite a green build.
    data = glob(
        [
            "utils/*.py",
            "**/*.yml",
        ],
        exclude = ["tools/g3_mirror_probe.py"],
    ),
    main = "tools/g3_mirror_probe.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/etils/epath",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/scamper:wandb_mock",
        "//pyglib:gfile",
    ],
)

# Can TFDS read ImageNet off Colossus? The `knn_full` eval runs ONCE, at the
# very end of a 57-hour stage-2, so a wrong data_dir is discovered at the
# most expensive possible moment. This answers it in seconds, without torch,
# JAX or the model graph -- see the file's docstring for what it checks.
py_binary(
    name = "g3_knn_tfds_probe",
    srcs = ["tools/g3_knn_tfds_probe.py"],
    main = "tools/g3_knn_tfds_probe.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [
        # The CNS filesystem registration. TFDS reads through
        # tensorflow.io.gfile, which only knows the /cns/ prefix if something
        # in the binary registers it -- that is exactly what this probe is
        # here to prove rather than assume.
        "//file/colossus/public:cns",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets",
    ],
)

# The auto-resume decision, tested off Borg.
#
# It is a PURE function over a directory listing, which is why it can be tested
# here at all -- and why it must be: the precedence bug it now covers (a warm
# start being re-applied on every Borg task restart, colliding with the
# checkpoint the previous attempt wrote) killed a production run after 11
# restarts, and cost ~3 h of v6p-64 to observe. The test runs in seconds.
# A py_binary, not a py_test: `:main` is itself a py_binary and strict deps
# forbid depending on one from a test (go/py-strict-deps). Every other probe
# target here has the same shape, and the file runs its own cases and exits
# non-zero on failure, so `blaze run` is a complete verdict.
py_binary(
    name = "test_autoresume",
    srcs = ["tests/test_autoresume.py"],
    data = glob(
        [
            "**/*.py",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
        ],
        exclude = ["tests/test_autoresume.py"],
    ),
    main = "tests/test_autoresume.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [":main"],
)

# The metrics tracker's device-side accumulation, checked for VALUE equality
# against the old host-side version. A performance fix that moves a number is
# worse than the slowness it cures, and this one touches every logged metric.
py_binary(
    name = "g3_metrics_tracker_probe",
    srcs = ["tools/g3_metrics_tracker_probe.py"],
    data = glob(["utils/*.py"]),
    main = "tools/g3_metrics_tracker_probe.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [
        # logging_util imports PIL and the wandb mock at module scope.
        "//third_party/py/PIL:pil",
        "//third_party/py/absl/logging",
        "//third_party/py/etils/epath",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/scamper:wandb_mock",
        "//pyglib:gfile",
    ],
)

# Which mesh does a real slice get? get_mesh() falls back to a flat 1-D debug
# mesh for an unknown TPU kind, silently, and a 1-D mesh under HSDP shards
# every parameter across all devices -- slow, never wrong, so nothing fails.
py_binary(
    name = "g3_mesh_probe",
    srcs = ["tools/g3_mesh_probe.py"],
    data = glob(["utils/*.py"]),
    main = "tools/g3_mesh_probe.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/absl:app",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

# Does configs/remote_run_eval_config.yml resolve, and carry the 17 benchmarks,
# INSIDE a Blaze binary? A plain-interpreter run cannot answer that:
# `configs/load_config.py::_find_config_yml` takes a DIFFERENT branch under
# runfiles ($GOOGLEBASE / sys.path entries), and that helper exists precisely
# because the Borg lookup differs from the workstation one. Verified 6/6 under
# `blaze run` before the eval-only launch.
py_binary(
    name = "test_eval_only_config",
    srcs = ["tests/test_eval_only_config.py"],
    data = glob(
        [
            "**/*.py",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
        ],
        exclude = ["tests/test_eval_only_config.py"],
    ),
    main = "tests/test_eval_only_config.py",
    strict_deps = False,
    tags = ["nostrictdeps"],
    deps = [":main"],
)
