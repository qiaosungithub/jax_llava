"""The cell -> metro -> CNS-cell routing table, and that it fails closed.

`utils/g3_env.py` is the single place that decides which storage a job reads
and writes. A wrong answer here is not a crash, it is a job quietly streaming
200 GiB across a continent for hours until the utilisation pruner kills it --
so the interesting assertions are the ones about REFUSING to answer.

Imports nothing heavy: g3_env touches no filesystem and no JAX by design.
"""

import os

import pytest

from utils import g3_env


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    """The overrides short-circuit resolution; clear them so the table is what
    is under test rather than whatever this shell happens to export."""
    for name in ('JAX_LLAVA_DATA_ROOT', 'JAX_LLAVA_CNS_CELL', 'JAX_LLAVA_ZONE',
                 'CHECKPOINT_BUCKET', 'BORG_CELL', 'BORG_PHYSICAL_CELL',
                 'BORG_TASK_HANDLE'):
        monkeypatch.delenv(name, raising=False)


# --- the move itself -------------------------------------------------------

def test_go_resolves_to_the_go_d_replica():
    """The point of the move: a job in cell `go` reads go-d, not yucmhcg-d."""
    assert g3_env.cns_data_root('go') == '/cns/go-d/home/qiaos/data'


def test_go_and_yucmhcg_are_the_same_metro():
    """Both cmh, so this is a quota move, not a locality move."""
    assert g3_env.metro_of_cell('go') == 'cmh'
    assert g3_env.metro_of_cell('yucmhcg') == 'cmh'


def test_cmh_prefers_go_d_but_keeps_yucmhcg_d_reachable():
    roots = g3_env.cns_data_roots('go')
    assert roots[0] == '/cns/go-d/home/qiaos/data'
    assert '/cns/yucmhcg-d/home/qiaos/data' in roots


def test_a_job_in_yucmhcg_still_resolves_the_same_metro_replicas():
    """Existing checkpoints and any job pinned to the old cell keep working."""
    assert g3_env.cns_data_roots('yucmhcg')[0].startswith('/cns/go-d/')


# --- fail closed -----------------------------------------------------------

def test_unknown_cell_raises_rather_than_guessing():
    with pytest.raises(ValueError) as excinfo:
        g3_env.cns_data_root('nosuchcell')
    assert 'refusing to guess' in str(excinfo.value).lower()


def test_ske_has_no_data_root_and_says_which_metros_do():
    """ske is a compute-only metro: 500 GiB personal ceiling, no group quota.

    It must NOT silently resolve to a cmh root when there is no
    $CHECKPOINT_BUCKET to make the cross-metro decision explicit.
    """
    with pytest.raises(ValueError) as excinfo:
        g3_env.cns_data_roots('yuskedq')
    message = str(excinfo.value)
    assert 'ske' in message
    assert 'yuskedq' in message


def test_excluded_cells_are_absent_from_the_routing_table():
    """The cells with no group registration must never be a destination."""
    for cell in ('yuskedq-d', 'yucbfpv-d', 'yuchspe-d'):
        assert cell not in g3_env._CNS_DATA_ROOTS
        for cells in g3_env._METRO_TO_CNS_CELLS.values():
            assert cell not in cells


def test_ske_is_not_in_the_cns_cell_table():
    assert 'ske' not in g3_env._METRO_TO_CNS_CELLS
    assert 'ske' not in g3_env._METRO_TO_REGION


# --- table self-consistency ------------------------------------------------
# These are the invariants an edit is most likely to break: adding a metro in
# one table and forgetting the other produces a KeyError or a silent ().

def test_every_metro_with_cns_cells_has_a_region():
    for metro in g3_env._METRO_TO_CNS_CELLS:
        assert metro in g3_env._METRO_TO_REGION, metro


def test_every_metro_with_cns_cells_is_reachable_from_some_cell():
    reachable = set(g3_env._CELL_TO_METRO.values())
    for metro in g3_env._METRO_TO_CNS_CELLS:
        assert metro in reachable, f'{metro} has storage but no compute cell'


def test_every_registered_data_root_lives_in_the_cell_that_names_it():
    for cell, root in g3_env._CNS_DATA_ROOTS.items():
        assert root.startswith(f'/cns/{cell}/'), (cell, root)


def test_every_data_root_cell_is_listed_under_some_metro():
    listed = {c for cells in g3_env._METRO_TO_CNS_CELLS.values() for c in cells}
    for cell in g3_env._CNS_DATA_ROOTS:
        assert cell in listed, f'{cell} has a root but no metro'


def test_metro_to_region_matches_the_depot_table():
    """//production/borg/cloud_iam/slicer_regions/slicer_metros.pi, read from
    the depot at the time of writing. Pinned here so an edit that invents a
    region fails a test rather than a job."""
    assert g3_env._METRO_TO_REGION == {
        'cmh': 'us-east5',
        'cbf': 'us-central1',
        'tul': 'us-central2',
        'dfw': 'us-south1',
        'grq': 'europe-west4',
        'phx': 'us-west8',
    }


def test_region_of_cell_is_none_for_a_cell_we_cannot_place():
    """None means 'cannot tell', which callers must treat as fatal."""
    assert g3_env.region_of_cell('nosuchcell') is None
    assert g3_env.region_of_cell('yuskedq') is None   # known cell, unknown region
    assert g3_env.region_of_cell('go') == 'us-east5'


# --- overrides -------------------------------------------------------------

def test_explicit_data_root_wins(monkeypatch):
    monkeypatch.setenv('JAX_LLAVA_DATA_ROOT', '/cns/go-d/home/qiaos/data2')
    assert g3_env.cns_data_roots('yuskedq') == ('/cns/go-d/home/qiaos/data2',)


def test_explicit_cns_cell_must_be_registered(monkeypatch):
    monkeypatch.setenv('JAX_LLAVA_CNS_CELL', 'yuskedq-d')
    with pytest.raises(ValueError) as excinfo:
        g3_env.cns_data_roots('yuskedq')
    assert 'no registered data root' in str(excinfo.value)


def test_explicit_cns_cell_go_d_resolves(monkeypatch):
    monkeypatch.setenv('JAX_LLAVA_CNS_CELL', 'go-d')
    assert g3_env.cns_data_roots('yuskedq') == ('/cns/go-d/home/qiaos/data',)


def test_cns_cell_of_path_round_trips():
    assert g3_env.cns_cell_of_path('/cns/go-d/home/qiaos/data') == 'go-d'
    assert g3_env.cns_cell_of_path('gs://bucket/x') is None


def test_cns_cells_for_zone_finds_cmh():
    assert g3_env.cns_cells_for_zone('us-east5') == ('go-d', 'yucmhcg-d')


# --- the cross-metro fallback stays loud and stays bounded -----------------

def test_cross_metro_fallback_requires_an_explicit_bucket(monkeypatch, capsys):
    """A job in ske with --bucket naming a cmh root may read across metros --
    that is a human decision made at submit time -- but it must announce it."""
    monkeypatch.setenv('CHECKPOINT_BUCKET', '/cns/go-d/home/qiaos/jax_llava')
    roots = g3_env.cns_data_roots('yuskedq')
    assert roots == ('/cns/go-d/home/qiaos/data',)
    assert 'CROSS-METRO DATA READ' in capsys.readouterr().out


def test_cross_metro_fallback_refuses_an_unregistered_bucket_cell(monkeypatch):
    monkeypatch.setenv('CHECKPOINT_BUCKET', '/cns/yuskedq-d/home/qiaos/jax_llava')
    with pytest.raises(ValueError):
        g3_env.cns_data_roots('yuskedq')


# --- models ----------------------------------------------------------------

def test_model_roots_prefer_go_d_with_a_working_fallback():
    roots = g3_env.cns_model_roots('go')
    assert roots[0] == '/cns/go-d/home/qiaos/models'
    assert '/cns/yucmhcg-d/home/qiaos/models' in roots
