#!/bin/bash
# Fence for the VRAM admission gate: exllamav3/generator/admission.py and its call site in
# Generator.iterate_start_jobs.
#
# Arms (law 12/21 - a fence that has not been seen to fire is not trusted):
#   GREEN  the landed tree: tier boundaries exact, margin arithmetic exact, gate wired at
#          the admission point, refusal on the engine's queue path
#   RED-a  wiring read from the pristine dsh-nobcast generator.py: the gate is not there,
#          so exactly the wiring tests must fail (a "failed" summary, not an import crash)
#   RED-b  the 8192-token ceiling poisoned to 8191 in a temp copy: the boundary test must
#          bite while the decode-exemption and purity tests still pass
#
# Any failure for any other reason is a fence failure. CPU only: no GPU, no torch, no
# model load, no wheel.
set -euo pipefail
trap 'echo "check_admission_gate.sh:${LINENO}: ${BASH_COMMAND}" >&2' ERR

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST="tests/generator/test_admission_gate.py"
PY="${PY:-/usr/bin/python3}"
BASE_REF="${BASE_REF:-dsh-nobcast}"
SCRUB=(env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=)
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
BAD=0

banner() { printf '\n== %s\n' "$*"; }
die() { printf 'FENCE FAIL: %s\n' "$*" >&2; BAD=1; }

# Run one arm; $1 = output file, rest = extra env assignments (may be empty)
arm() {
    local out="$1"; shift
    ( cd "${REPO}" && env "$@" "${SCRUB[@]}" "${PY}" -m pytest --noconftest -q -rf "${TEST}" \
        ) >"${out}" 2>&1 || true
}

collected() {
    local out="${WORK}/collect.txt"
    ( cd "${REPO}" && "${SCRUB[@]}" "${PY}" -m pytest --noconftest -q --collect-only "${TEST}" \
        ) >"${out}" 2>&1 || true
    grep -oE '[0-9]+ tests? collected' "${out}" | grep -oE '^[0-9]+' | head -1
}

failed_list() { grep -oE '^FAILED [^ ]+::[A-Za-z0-9_]+' "$1" | sed 's|.*::||' | LC_ALL=C sort -u || true; }
count_word() { grep -oE "[0-9]+ $1" "$2" | tail -1 | grep -oE '^[0-9]+' || true; }
summary() { grep -E '[0-9]+ (passed|failed|error)' "$1" | tail -1 || true; }

TOTAL="$(collected)"
[ -n "${TOTAL}" ] && [ "${TOTAL}" -gt 0 ] || { echo "FENCE FAIL: no tests collected" >&2; exit 1; }
echo "positive control: ${TOTAL} tests collected from ${TEST}"

# ---------------------------------------------------------------- GREEN
banner "ARM GREEN (landed tree)"
arm "${WORK}/green.txt"
raw="$(summary "${WORK}/green.txt")"
printf 'raw summary: %s\n' "${raw}"
[ -n "${raw}" ] || die "GREEN: no pytest summary line (the run did not finish)"
green_passed="$(count_word passed "${WORK}/green.txt")"
green_failed="$(count_word failed "${WORK}/green.txt")"
green_bites="$(failed_list "${WORK}/green.txt")"
[ -n "${green_passed}" ] || die "GREEN: cannot read a passed count"
if [ -z "${green_bites}" ] && [ "${green_passed}" -eq "${TOTAL}" ]; then
    echo "GREEN ok: ${green_passed}/${TOTAL} passed, no failures"
else
    die "GREEN: expected ${TOTAL} passed and no failures, got passed=${green_passed:-0} failed=${green_failed:-0}"
    printf '%s\n' "${green_bites}" | sed '/^$/d;s/^/  failed: /'
fi

# ---------------------------------------------------------------- RED-a
banner "ARM RED-a (wiring against pristine ${BASE_REF} generator.py)"
PRISTINE="${WORK}/generator_pristine.py"
git -C "${REPO}" show "${BASE_REF}:exllamav3/generator/generator.py" >"${PRISTINE}"
if grep -q 'admits(' "${PRISTINE}"; then
    die "RED-a: the base generator.py already calls admits() - the arm cannot prove anything"
else
    echo "sabotage landed: ${BASE_REF} generator.py contains no admits() call"
fi
arm "${WORK}/reda.txt" "EXL3_GENERATOR_PATH=${PRISTINE}"
printf 'raw summary: %s\n' "$(summary "${WORK}/reda.txt")"
reda_failed="$(failed_list "${WORK}/reda.txt")"
if [ -z "${reda_failed}" ]; then
    die "RED-a: expected failures and got none"
elif printf '%s\n' "${reda_failed}" | grep -qv '^test_wiring_'; then
    die "RED-a: wrong reason - non-wiring tests failed too: $(printf '%s ' ${reda_failed})"
elif grep -qiE '[0-9]+ error' "${WORK}/reda.txt"; then
    die "RED-a: collection/import error instead of a clean failed summary"
else
    reda_passed="$(count_word passed "${WORK}/reda.txt")"
    reda_nfailed="$(printf '%s\n' "${reda_failed}" | grep -c . )"
    if [ "${reda_passed}" -eq $(( TOTAL - reda_nfailed )) ]; then
        echo "RED-a ok: ${reda_nfailed} wiring tests failed (${reda_failed}), ${reda_passed} still passed"
    else
        die "RED-a: counts do not reconcile (passed=${reda_passed:-0}, failed=${reda_nfailed}, collected=${TOTAL})"
    fi
fi

# ---------------------------------------------------------------- RED-b
banner "ARM RED-b (boundary poison: 8192 -> 8191)"
POISON="${WORK}/admission_poisoned.py"
sed 's/(8192, 20)/(8191, 20)/' "${REPO}/exllamav3/generator/admission.py" >"${POISON}"
if ! grep -q '(8191, 20)' "${POISON}" || grep -q '(8192, 20)' "${POISON}"; then
    die "RED-b: the poison did not land in ${POISON}"
else
    echo "sabotage landed: tier table now reads (8191, 20)"
fi
arm "${WORK}/redb.txt" "EXL3_ADMISSION_PATH=${POISON}"
printf 'raw summary: %s\n' "$(summary "${WORK}/redb.txt")"
redb_failed="$(failed_list "${WORK}/redb.txt")"
if [ -z "${redb_failed}" ]; then
    die "RED-b: the boundary test did not bite"
elif ! printf '%s\n' "${redb_failed}" | grep -q '^test_cost_table_boundaries$'; then
    die "RED-b: wrong reason - the boundary test passed: $(printf '%s ' ${redb_failed})"
elif printf '%s\n' "${redb_failed}" | grep -q '^test_decode_never_blocked$'; then
    die "RED-b: decode exemption also broke - the poison moved more than the boundary"
elif grep -qiE '[0-9]+ error' "${WORK}/redb.txt"; then
    die "RED-b: collection/import error instead of a clean failed summary"
else
    echo "RED-b ok: boundary bit ($(printf '%s ' ${redb_failed})) with decode/purity tests intact"
fi

banner "VERDICT"
if [ "${BAD}" -eq 0 ]; then
    echo "check_admission_gate.sh: all arms ok (GREEN + 2 RED arms, each for its own reason)"
else
    echo "check_admission_gate.sh: RED" >&2
fi
exit "${BAD}"