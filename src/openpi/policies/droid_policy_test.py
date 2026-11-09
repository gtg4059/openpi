import numpy as np

from openpi.policies.droid_policy import AttachLiveCri


class _FakeCriFilterSolver:
    def __init__(self) -> None:
        self.calls = 0

    def run_cri_filter(self, q, qd):
        del q, qd
        self.calls += 1
        return {"cri_pre": np.full(9, 0.42, dtype=np.float32)}


def test_attach_live_cri_uses_delayed_run_cri_filter_pre():
    attach = AttachLiveCri()
    attach._state.solver = _FakeCriFilterSolver()
    obs = {
        "observation/joint_position": np.zeros(7, dtype=np.float64),
        "observation/joint_velocity": np.ones(7, dtype=np.float64),
    }

    first = attach(obs)
    second = attach(obs)

    assert attach._state.solver.calls == 2
    np.testing.assert_allclose(first["observation/cri"], np.zeros(9, dtype=np.float32))
    np.testing.assert_allclose(second["observation/cri"], np.full(9, 0.42, dtype=np.float32))


def test_attach_live_cri_does_not_call_compute_when_cri_present():
    attach = AttachLiveCri()
    attach._state.solver = _FakeCriFilterSolver()
    given = np.arange(9, dtype=np.float32)
    out = attach({"observation/cri": given, "observation/joint_position": np.zeros(7)})

    assert attach._state.solver.calls == 0
    np.testing.assert_array_equal(out["observation/cri"], given)
