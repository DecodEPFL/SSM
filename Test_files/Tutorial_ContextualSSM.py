r"""
Tutorial: context-enriched, L2-stable Performance Boosting (``ContextualDeepSSM``)
==================================================================================

One operator, ``ContextualDeepSSM``, maps a reconstructed disturbance ``w_hat``
and an arbitrary context signal ``z`` to a control correction ``u`` while
preserving the closed-loop L2 guarantee of the underlying ``DeepSSM`` core:

    u_t = A(w_hat, z)_t  (x)  core([ w_hat_t ; Pi(z)_t ])_t      (gates live inside core)
          \____ mixer ___/        \core/  \_filter_/
            Port A                  Port B (additive)     Port C is inside the core blocks

Three ports are toggled independently through ``context_modes``:

  * "mixer" (Port A): a uniformly *bounded* matrix A_t in R^{d_output x d_features}
    multiplies the core features e_t.  bounded x l2 = l2, unconditionally -- even
    when z is the system state.  It is a ROUTER: its output is proportional to the
    disturbance feature e(w_hat), so it vanishes when the disturbance is small.

  * "input" (Port B): context is projected into an l2 sequence Pi(z) and
    concatenated to the disturbance before the core.  This is the only port that
    SYNTHESISES control from context alone, so it is the one that helps when the
    disturbance is small and that makes context-dependent behaviour learnable.

  * "gate" (Port C): per-block sigmoid gates in [0,1] attenuate the SSM/FF
    branches.  Safe because gates only attenuate the certified branch gains.

Context projections for the "input" path (``context_filter``):
    finite_horizon  1 on t<T then 0          (loss-free on the horizon)
    taper           flat then cosine roll-off (smooth finite_horizon)
    difference      z_t - z_{t-1}            (l2 for switching/BV context; loss-free)
    exponential     rho**t                   (anytime guarantee)
    polynomial      (t+1)**-power            (anytime guarantee)
    none            identity                 (sys-ID / finite-gain; NOT l2)

Routing rule: exogenous context (reference, obstacle position) -> "input";
endogenous/in-loop context (system state) -> "mixer" (unconditionally safe).

Certificates (need a prescribed ``gamma``):
    certified_gain_bound()      disturbance->control l2 gain  (core_gain * ||A||_inf)
    context_offset_bound(amp)   additive context bias term    (inf for difference/none)
    additive_channel_gain()     context->u gain (small-gain check for endogenous z)
    gain_diagnostics()          full breakdown

Run ``python Test_files/Tutorial_ContextualSSM.py`` to see all of this in action.
"""

from __future__ import annotations

import torch

try:  # runnable whether the package is installed or used from a source checkout
    from neural_ssm.ssm import ContextualDeepSSM
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.neural_ssm.ssm import ContextualDeepSSM


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    torch.manual_seed(0)

    # --- a tiny synthetic PB scenario --------------------------------------
    # disturbance w_hat : a short velocity-channel burst on t in [0,5), then
    #                     silence.  Finitely supported  =>  it is an l2 sequence.
    # context z         : a piecewise-constant "gate" that switches at t=40 and
    #                     t=90, i.e. well AFTER the disturbance has died out.
    B, T, n, q, m = 1, 120, 4, 1, 2          # batch, horizon, dim(w_hat), dim(z), dim(u)
    w = torch.zeros(B, T, n)
    w[:, 0:5, 2:] = 0.6 * torch.randn(B, 5, 2)   # burst on the two velocity channels
    z = torch.zeros(B, T, q)
    z[:, 40:90] = 1.0
    z[:, 90:] = -0.5
    quiet = slice(40, 100)                        # NO disturbance here, but two switches

    # Shared certified L2 core: prescribed gain cap (gamma) + certified
    # parametrization (l2n / tv / tvc) and a Lipschitz feed-forward.
    core = dict(param="l2n", ff="MBLIP", gamma=1.0, d_model=24, d_state=24, n_layers=2)

    # =======================================================================
    banner("Port A -- mixer (a ROUTER): output tracks the disturbance feature")
    mixer = ContextualDeepSSM(
        n, q, m, context_modes=("mixer",), d_features=16, mixer_bound=4.0, **core
    ).eval()
    with torch.no_grad():
        u_mix, _ = mixer(w, z, mode="loop")
    print(f"||u|| during burst  t in [0,5)   : {u_mix[:, 0:5].norm():.4f}")
    print(f"||u|| at switches   t in [40,100): {u_mix[:, quiet].norm():.4f}   <- decays with w_hat")

    # =======================================================================
    banner("Port B -- input + difference (a DRIVER): acts on context switches")
    driver = ContextualDeepSSM(
        n, q, m, context_modes=("input",), context_filter="difference", **core
    ).eval()
    with torch.no_grad():
        u_drv, _ = driver(w, z, mode="loop")
    print(f"||u|| during burst  t in [0,5)   : {u_drv[:, 0:5].norm():.4f}")
    print(f"||u|| at switches   t in [40,100): {u_drv[:, quiet].norm():.4f}   <- driven by the gate")

    # =======================================================================
    banner("The small-disturbance limit: feed w_hat == 0")
    zero = torch.zeros_like(w)
    with torch.no_grad():
        um, _ = mixer(zero, z, mode="loop")
        ud, _ = driver(zero, z, mode="loop")
    print(f"mixer  max|u| with w_hat=0 : {um.abs().max():.3e}   (router is exactly silent)")
    print(f"driver max|u| with w_hat=0 : {ud.abs().max():.3e}   (driver still acts on context)")

    # =======================================================================
    banner("Certificates (require a prescribed gamma)")
    print(f"mixer  certified w->u gain : {mixer.certified_gain_bound().item():.3f}  (= gamma * mixer_bound)")
    print(f"driver certified w->u gain : {driver.certified_gain_bound().item():.3f}  (= gamma, no mixer)")

    # A finite-horizon driver also yields a finite additive-bias bound.
    fh = ContextualDeepSSM(
        n, q, m, context_modes=("input",), context_filter="finite_horizon", horizon=T, **core
    ).eval()
    amp = float(z.abs().amax())                   # sup_t ||z_t||
    with torch.no_grad():
        u0, _ = fh(zero, z, mode="loop")          # w_hat=0 => output is the pure context drive
    print(f"finite-horizon offset bound: {fh.context_offset_bound(amp):.3f}")
    print(f"  realised ||u|| (w_hat=0) : {u0.norm():.3f}   <= offset bound")
    print(f"difference offset bound    : {driver.context_offset_bound(amp)}   (no fixed l2 window)")

    # =======================================================================
    banner("Verify the certified bound holds on a NONZERO disturbance")
    # The guarantee is the affine bound  ||u||_2 <= gain * ||w_hat||_2 + offset.
    with torch.no_grad():
        # mixer: no additive context, so offset = 0  =>  ||u|| <= gain * ||w_hat||.
        u_m, _ = mixer(w, z, mode="loop")
        g_m = mixer.certified_gain_bound().item()
        rhs_m = g_m * w.norm().item()
        assert u_m.norm().item() <= rhs_m + 1e-4
        print(f"mixer : ||u||={u_m.norm():.3f} <= gain*||w_hat|| "
              f"= {g_m:.2f}*{w.norm():.3f} = {rhs_m:.3f}   OK")

        # finite-horizon driver: full affine bound with the additive offset.
        u_f, _ = fh(w, z, mode="loop")
        g_f = fh.certified_gain_bound().item()
        off = fh.context_offset_bound(amp)
        rhs_f = g_f * w.norm().item() + off
        assert u_f.norm().item() <= rhs_f + 1e-4
        print(f"driver: ||u||={u_f.norm():.3f} <= gain*||w_hat||+offset "
              f"= {g_f:.2f}*{w.norm():.3f}+{off:.2f} = {rhs_f:.3f}   OK")

    # =======================================================================
    banner("Combine all three ports (+ sys-ID note)")
    full = ContextualDeepSSM(
        n, q, m,
        context_modes=("input", "gate", "mixer"),
        context_filter="taper", horizon=T, context_filter_ramp=20,
        d_features=16, mixer_bound=4.0, **core,
    ).eval()
    with torch.no_grad():
        u, _, aux = full(w, z, mode="loop", return_aux=True)
    diag = full.gain_diagnostics()
    print(f"output {tuple(u.shape)} | mixer A {tuple(aux['mixer'].shape)} | "
          f"gates/block {len(aux['context_gates'])}")
    print("diagnostics:", {k: diag[k] for k in
                           ("context_modes", "matrix_norm_bound", "certified_gain_bound")})
    print("\nSystem identification / learning: use context_filter='none' "
          "(finite-gain prior; the output need not vanish).")


if __name__ == "__main__":
    main()
