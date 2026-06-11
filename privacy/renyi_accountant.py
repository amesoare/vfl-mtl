"""privacy/renyi_accountant.py — Per-task Rényi DP accounting for VFL-MTL (Mironov 2017)."""

from __future__ import annotations

from opacus.accountants import RDPAccountant

_TASKS = ("ihm", "decomp", "pheno")
_COUPLING_KEYS = (
    "grad_sim_ihm_decomp",
    "grad_sim_ihm_pheno",
    "grad_sim_decomp_pheno",
)


class RenyiAccountant:

    def __init__(self) -> None:
        self._accountants: dict[str, RDPAccountant] = {
            t: RDPAccountant() for t in _TASKS
        }
        self._grad_sim_history: list[dict[str, float]] = []


    # Stepping

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
        task: str | None = None,
    ) -> None:
        """Step accountant(s) by num_steps; task=None steps all (uniform mode)."""
        targets = _TASKS if task is None else (task,)
        for t in targets:
            for _ in range(num_steps):
                self._accountants[t].step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                )

    def step_stratified(
        self,
        sigma_map: dict[str, float],
        sample_rate: float,
        num_steps: int = 1,
    ) -> None:
        """Step each task accountant with its own σ (stratified mode)."""
        for t, sigma in sigma_map.items():
            if t in self._accountants:
                for _ in range(num_steps):
                    self._accountants[t].step(
                        noise_multiplier=sigma,
                        sample_rate=sample_rate,
                    )


    # Querying

    def get_epsilon(self, delta: float = 1e-5) -> dict[str, float]:
        """Return {task: ε_k} at the given δ; nan for tasks with no history."""
        result: dict[str, float] = {}
        for t, acc in self._accountants.items():
            try:
                result[t] = float(acc.get_epsilon(delta=delta))
            except Exception:
                result[t] = float("nan")
        return result


    # Gradient coupling matrix

    def log_grad_sim(self, grad_sim_dict: dict[str, float]) -> None:
        """Accumulate one round of gradient cosine-similarity values from compute_task_gradient_similarity()."""
        self._grad_sim_history.append(
            {k: v for k, v in grad_sim_dict.items() if k in _COUPLING_KEYS}
        )

    def cross_task_coupling_matrix(self) -> dict[str, float]:
        """Mean gradient cosine-similarity (ρ proxy) across logged rounds; {} if none logged."""
        if not self._grad_sim_history:
            return {}

        result: dict[str, float] = {}
        for key in _COUPLING_KEYS:
            vals = [
                d[key] for d in self._grad_sim_history
                if key in d and d[key] == d[key]   # skip NaN
            ]
            result[key] = float(sum(vals) / len(vals)) if vals else float("nan")
        return result

    def coupling_epsilon_inflation(self, delta: float = 1e-5) -> float:
        """Additive ε inflation from multi-task composition: Σ_k ε_k − max_k ε_k."""
        eps = self.get_epsilon(delta=delta)
        valid = [v for v in eps.values() if v == v]    # skip NaN
        if not valid:
            return 0.0
        return float(sum(valid) - max(valid))



    @property
    def n_logged_rounds(self) -> int:
        """Number of rounds for which gradient similarity has been logged."""
        return len(self._grad_sim_history)

    def __repr__(self) -> str:
        try:
            eps = self.get_epsilon()
            eps_str = ", ".join(f"{t}={v:.3f}" for t, v in eps.items())
        except Exception:
            eps_str = "not stepped"
        return f"RenyiAccountant(epsilon={{{eps_str}}}, logged_rounds={self.n_logged_rounds})"
