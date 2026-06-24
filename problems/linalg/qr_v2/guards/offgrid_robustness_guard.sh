#!/usr/bin/env bash
# QR v2 off-grid-n robustness guard (context-corruption-cascade detector).
#
# WHAT IT CATCHES — a size-dispatching kernel that is only correct on the exact
# matrix sizes the PUBLIC benchmark exercises ({32,176,352,512,1024,2048,4096})
# and FAULTS / HARD-ASSERTS / RETURNS GARBAGE on any OTHER n. The qr_v2 custom
# kernels do row-strided vectorized loads (build_V_kernel<half>,
# fill_above_panel_R_tiled, the bf16 panel kernels) that are only aligned when
# n%8==0, and the tiny/blocked paths assert exact sizes (blocked_qr_tiny needs
# n==32). So a held-out n that lands off-grid hits one of three failure modes,
# all measured here on the unguarded kernel:
#   - cudaErrorMisalignedAddress  (n%8!=0 in a vectorized range, e.g. 300, 351),
#   - a hard RuntimeError/assert   (n<32, e.g. 16: "blocked_qr_tiny: only n==32"),
#   - silent garbage               (n in the (352,512) gap, e.g. 400: residual ~600).
# The board-confirmed fix routes ANY n not on the exact verified grid to
# torch.geqrf; an honest kernel that does so passes this guard trivially.
#
# WHY THIS IS A CONTEXT-CORRUPTION CASCADE, NOT A LOCAL ONE-SHOT FAILURE — eval.py
# runs EVERY test shape through ONE persistent worker process (Pool(1), spawn),
# so all shapes share ONE long-lived CUDA context. When an off-grid shape throws
# a CUDA fault MID-RUN, that fault corrupts the shared context: every SUBSEQUENT
# shape's CUDA call then blocks or errors, and the whole run wall-clocks to the
# leaderboard watchdog (the secret run was observed timing out at an exact 360s).
# So a single off-grid n does not fail in isolation -- it takes the entire run
# down with it, including the on-grid shapes that would otherwise pass.
#
# WHY EVAL / THE PUBLIC TEST SHAPES MISS IT — validate.sh runs eval.py test only
# over task.yml's PUBLIC tests:, whose every n is already on the supported grid.
# The HELD-OUT secret test set draws shapes the public set never exercises --
# off-grid n (n%8!=0, n<32, the (352,512) gap) and larger/odd batches. So a clean
# local validate over the public shapes did NOT imply a secret-safe submission:
# this exact class shipped green locally and then timed out on the secret set.
# This guard closes that gap by exercising the off-grid n the public shapes omit.
#
# HOW — drives the SHIPPED eval.py test mechanism (no bespoke kernel-calling: a
# spec file of off-grid n is fed to eval.py exactly as validate.sh feeds the
# public specs, so reference.py's real correctness gates apply unchanged). The
# spec interleaves off-grid n {300,351,16,400,2050} with on-grid n=512 sentinels
# and ENDS on an on-grid n=512 -- all in ONE eval process / ONE CUDA context, the
# exact cascade condition. If the guarded routing holds, every shape passes and
# eval prints `check: pass`. If a kernel faults on an off-grid shape, one of:
#   (a) eval exits non-zero (the fault propagated as a Python exception), or
#   (b) the off-grid shape reports `check: fail` (garbage residual), or
#   (c) the context is poisoned and a later sentinel hangs -> the outer timeout
#       fires and turns the 360s cascade into a fast, clear local failure,
# and the guard FAILS LOUDLY (non-zero exit). Bounded-time by construction: the
# off-grid shapes route to geqrf on a fixed kernel and the n=512 sentinels use a
# tiny batch, so a healthy run is a few seconds; the timeout caps a sick one.
#
# BUILD-LOCK HYGIENE — eval.py's spawn worker re-imports submission, which
# triggers torch load_inline behind a per-build-dir baton lock. If the outer
# timeout kills the worker mid-import (case (c)), a stale lock can be left behind
# and would hang the NEXT validate's import for the full baton wait. So this guard
# clears any stale (unheld) torch build lock both BEFORE and AFTER the run, the
# same way harness/build.sh does -- converting a would-be silent hang into a
# self-healing fast failure here.
#
# BOUNDARY — it asserts the kernel SURVIVES off-grid n without taking the run
# down; it does not exhaustively prove correctness at every possible n. It checks
# a representative set spanning each failure mode (misaligned, assert, gap,
# high-range n%8!=0). A kernel correct on these five but broken on some other
# off-grid n is not caught here -- but the board-confirmed fix (route every
# off-grid n to geqrf) is correct for ALL of them at once, so passing this set
# with that fix in place is meaningful.
#
# ENABLED: this guard runs on every validate (harness/validate.sh globs and runs
# guards/*.sh after the public test shapes pass). It spends a few seconds of real
# GPU time -- accepted as the price of turning an invisible secret-only timeout
# into a loud local validation failure.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_DIR/harness/env.sh" "$@"

# --- stale torch build-lock hygiene (mirror of harness/build.sh) -------------
# Remove a torch load_inline baton lock only when NO process is holding it (a
# crash/timeout left it behind). Never touch a lock another live build holds.
clear_unheld_torch_locks() {
    local lock ext_dir removed=0
    [ -d "$TORCH_EXTENSIONS_DIR" ] || return 0
    while IFS= read -r -d '' lock; do
        ext_dir="$(dirname "$lock")"
        if fuser "$lock" >/dev/null 2>&1; then
            echo "offgrid guard: keeping active torch build lock $lock" >&2
            continue
        fi
        if command -v lsof >/dev/null 2>&1 && lsof +D "$ext_dir" >/dev/null 2>&1; then
            echo "offgrid guard: keeping busy torch build dir $ext_dir" >&2
            continue
        fi
        rm -f -- "$lock" && removed=$((removed + 1))
    done < <(find "$TORCH_EXTENSIONS_DIR" -mindepth 2 -maxdepth 2 -type f -name lock -print0 2>/dev/null)
    [ "$removed" -eq 0 ] || echo "offgrid guard: cleared $removed stale torch build lock(s)" >&2
}

# Off-grid n the PUBLIC test set never exercises, each measured to break an
# unguarded kernel (misaligned / assert / garbage), plus a high-range n%8!=0
# (2050) for forward coverage -- interleaved with on-grid n=512 sentinels and
# ENDING on an on-grid n=512, all in ONE eval process / ONE CUDA context so a
# fault's cascade reaches the trailing sentinel. Small batches keep it fast.
SPECFILE="$(mktemp)"
OUT="$(mktemp)"
ERR="$(mktemp)"
trap 'rm -f "$SPECFILE" "$OUT" "$ERR"' EXIT

cat > "$SPECFILE" <<'SPEC'
batch: 2; n: 512; cond: 2; seed: 32523
batch: 2; n: 300; cond: 1; seed: 99001
batch: 2; n: 512; cond: 2; seed: 32523
batch: 2; n: 351; cond: 1; seed: 99002
batch: 4; n: 16; cond: 1; seed: 99003
batch: 2; n: 400; cond: 1; seed: 99004
batch: 2; n: 2050; cond: 1; seed: 99005
batch: 2; n: 512; cond: 2; seed: 32523
SPEC

echo "off-grid-n robustness guard: driving eval.py test over off-grid n {300,351,16,400,2050}"
echo "  interleaved with on-grid n=512 sentinels, one eval process, ${OFFGRID_GUARD_TIMEOUT:-120}s cap"

cd "$PROBLEM_DIR" || exit 1

clear_unheld_torch_locks

# Bounded-time: a poisoned-context hang becomes a fast, clear failure here rather
# than a 360s watchdog timeout on the secret set. --preserve-status + --signal so
# a clean SIGTERM (rc 124/143) is distinguishable as "timed out".
GUARD_TIMEOUT="${OFFGRID_GUARD_TIMEOUT:-120}"
timeout --preserve-status --signal=TERM --kill-after=15s "$GUARD_TIMEOUT" \
    env POPCORN_FD=3 "$PYTHON" "$EVAL_PY" test "$SPECFILE" 3>"$OUT" 2>"$ERR"
rc=$?

# Always re-clear: if the timeout above killed the eval worker mid-import, it may
# have left a baton lock that would hang the next validate's build.
clear_unheld_torch_locks

echo "----- eval.py test output (off-grid sequence) -----"
cat "$OUT"
echo "---------------------------------------------------"

if [ "$rc" -eq 124 ] || [ "$rc" -eq 143 ]; then
    echo "off-grid-n robustness guard: FAILED -- eval HUNG over the off-grid sequence"
    echo "  (timed out after ${GUARD_TIMEOUT}s). This is the context-corruption cascade:"
    echo "  an off-grid n faulted mid-run, poisoned the shared CUDA context, and a later"
    echo "  shape blocked -- exactly the secret-set 360s timeout, caught locally. The fix is"
    echo "  to route every n not in (32,176,352,512,1024,2048,4096) to torch.geqrf."
    echo "----- eval stderr (tail) -----"; tail -20 "$ERR"; echo "------------------------------"
    exit 2
fi

if [ "$rc" -ne 0 ]; then
    echo "off-grid-n robustness guard: FAILED -- eval exited $rc on the off-grid sequence"
    echo "  An off-grid n faulted (e.g. cudaErrorMisalignedAddress for n%8!=0, or a size"
    echo "  assert for n<32) and crashed the run. Route every off-grid n to torch.geqrf."
    echo "----- eval stderr (tail) -----"; tail -20 "$ERR"; echo "------------------------------"
    exit 2
fi

if ! grep -qx "check: pass" "$OUT"; then
    echo "off-grid-n robustness guard: FAILED -- no 'check: pass' over the off-grid sequence"
    echo "  An off-grid n produced a bad QR factorization (e.g. garbage residual in the"
    echo "  (352,512) gap) so reference.py rejected it. Route every off-grid n to torch.geqrf."
    echo "----- eval stderr (tail) -----"; tail -20 "$ERR"; echo "------------------------------"
    exit 2
fi

echo "off-grid-n robustness guard: CLEAN (all off-grid n + on-grid sentinels passed)"
exit 0
