import os
import sys

import yaml

from configs.default import get_config as get_default_config


def _find_config_yml(mode_string):
    """Absolute path of `configs/<mode>_config.yml`, wherever this is running.

    `os.path.abspath(__file__)` is NOT enough, and the failure is Borg-only.
    `config_flags` loads this module by the path given on the command line --
    `--config=experimental/.../configs/load_config.py:g3_smoke` -- which is
    RELATIVE, so `__file__` is relative too and `abspath` resolves it against
    the current working directory. Locally that directory is the google3 root
    and everything works; on Borg it is the borglet's task directory, and the
    lookup lands on

        /export/hda3/borglet/local_ram_fs_dirs/0.qiaos_group_.../google3/
            experimental/.../configs/g3_smoke_config.yml

    which does not exist. `config_flags` then wraps the FileNotFoundError in
    its own deferred-error object, so the actual crash surfaces much later and
    somewhere else entirely -- at the first attribute access on FLAGS.config,
    inside `_apply_env_config_overrides`, as an opaque `_ReportError()`.
    Diagnosed from XID 277033539, whose import-crash marker carried the full
    chain.

    So try the candidates in order of trustworthiness and take the first that
    exists, rather than trusting any single one:
      1. `__file__` as given, if it is already absolute and present;
      2. $GOOGLEBASE, the runfiles root Borg exports -- authoritative there;
      3. every sys.path entry that ends in the package dir, which covers the
         PYTHONPATH the launcher sets;
      4. `abspath(__file__)`, the original behaviour, so nothing regresses.
    """
    rel = f"configs/{mode_string}_config.yml"
    here = __file__
    candidates = []
    if os.path.isabs(here):
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(here)), rel))
    googlebase = os.environ.get("GOOGLEBASE", "").strip()
    if googlebase:
        pkg = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0].strip()
        if pkg:
            candidates.append(os.path.join(googlebase, "google3", pkg, rel))
        # …and next to this module inside the runfiles tree.
        candidates.append(os.path.join(
            googlebase, "google3", os.path.dirname(os.path.dirname(here)), rel))
    for entry in sys.path:
        if entry:
            candidates.append(os.path.join(entry, rel))
    candidates.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(here))), rel))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not locate {rel}. __file__={here!r}, cwd={os.getcwd()!r}, "
        f"GOOGLEBASE={googlebase!r}. Tried: {candidates!r}"
    )


def get_config(mode_string):
    config_file = _find_config_yml(mode_string)
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    default_config = get_default_config()

    for k, v in config_dict.items():
        if isinstance(v, dict):
            default_config[k].update(v)
        else:
            default_config[k] = v

    # Backward-compatible alias: some yaml files used max_txt_length, while
    # input_pipeline reads max_txt_len.
    if 'max_txt_length' in default_config.dataset:
        default_config.dataset.max_txt_len = default_config.dataset.max_txt_length

    # if finetune: True is set in remote_run_config.yml, load finetune_config.yml instead
    if mode_string == "remote_run" and default_config['finetune']:
        return get_config("finetune")

    return default_config
