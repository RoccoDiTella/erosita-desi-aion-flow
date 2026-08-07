"""The stub switch must be honoured by EVERY encoder consumer, not just train.

`AIONFLOW_STUB_ENCODER=1` used to be read inside `run_train_multi` alone, so a
workstation could train a checkpoint and then not evaluate it: eval, posterior
structure and the HR quadrature each built the real `AIONTokenEncoder`
unconditionally and died on `import aion`. The eval half of the chain was
therefore first exercised on the cluster, after the expensive half had already
succeeded -- exactly the failure the stub exists to prevent.

These tests pin the two halves of the fix: `build_encoder` is the switch, and
the downstream scripts go through it.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stub_encoder import StubTokenEncoder, build_encoder, stub_requested  # noqa: E402

REPO = ROOT.parent
# The CLS-era consumers. `main.py` is the pre-CLS single-target path and builds
# a different encoder; it is not part of this contract.
CONSUMERS = ("scripts/eval_multitarget.py", "scripts/posterior_structure.py",
             "scripts/hr_from_joint.py")


@pytest.fixture()
def stub_on(monkeypatch):
    monkeypatch.setenv("AIONFLOW_STUB_ENCODER", "1")


def test_switch_reads_the_env_var(monkeypatch):
    for off in ("", "0", "false", "False"):
        monkeypatch.setenv("AIONFLOW_STUB_ENCODER", off)
        assert not stub_requested()
    monkeypatch.setenv("AIONFLOW_STUB_ENCODER", "1")
    assert stub_requested()


def test_build_encoder_returns_the_stub_on_cpu(stub_on, capsys):
    enc = build_encoder(num_cls=8, device=torch.device("cpu"), tag="eval")
    assert isinstance(enc, StubTokenEncoder)
    assert enc.cls_token.shape[0] == 8
    assert enc.cls_token.device.type == "cpu"
    # The warning is the whole point: a silent stub run produces a full table of
    # meaningless numbers that looks exactly like a real one.
    out = capsys.readouterr().out
    assert "AIONFLOW_STUB_ENCODER" in out and "meaningless" in out


def test_stub_state_round_trips_through_a_checkpoint(stub_on):
    """A stub-trained checkpoint must load back into a stub encoder.

    This is the join between the two halves of the chain: `run_train_multi`
    saves `encoder_trainable_state()`, and eval reloads it with
    `load_state_dict(..., strict=False)` while asserting nothing is unexpected.
    """
    trained = build_encoder(num_cls=8, device=torch.device("cpu"))
    with torch.no_grad():
        trained.cls_token.add_(1.0)
    fresh = build_encoder(num_cls=8, device=torch.device("cpu"))
    missing, unexpected = fresh.load_state_dict(
        trained.encoder_trainable_state(), strict=False)
    assert not unexpected, unexpected
    assert torch.allclose(fresh.cls_token, trained.cls_token)


@pytest.mark.skipif(importlib.util.find_spec("aion") is not None,
                    reason="aion installed, so the real encoder is buildable")
def test_missing_aion_names_the_escape_hatch(monkeypatch):
    monkeypatch.delenv("AIONFLOW_STUB_ENCODER", raising=False)
    with pytest.raises(ImportError, match="AIONFLOW_STUB_ENCODER"):
        build_encoder(num_cls=8, device=torch.device("cpu"))


@pytest.mark.parametrize("rel", CONSUMERS)
def test_consumers_do_not_construct_the_encoder_directly(rel):
    src = (REPO / rel).read_text()
    assert "build_encoder(" in src, f"{rel} does not go through build_encoder"
    assert not re.search(r"^\s*\w*\s*=?\s*AIONTokenEncoder\(", src, re.M), (
        f"{rel} constructs AIONTokenEncoder directly, so AIONFLOW_STUB_ENCODER "
        f"cannot reach it and this script is unrunnable without `aion`")


@pytest.mark.parametrize("rel", CONSUMERS)
def test_consumers_accept_a_device(rel):
    """No --device meant no CPU run, whatever the encoder did."""
    assert '"--device"' in (REPO / rel).read_text(), f"{rel} has no --device flag"
