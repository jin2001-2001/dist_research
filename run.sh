#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_torch_quota.sh -i <iface> -n <nnodes> -r <node_rank> [options] -- [D2D args...]

Required:
  -i IFACE          Socket/interface name (e.g., enp4s0f1)
  -n NNODES         Total number of nodes
  -r NODE_RANK      This node's rank (0..NNODES-1)

CPU limit (pick one; -c overrides -q if both provided):
  -q QUOTA_PCT      Percent of TOTAL machine CPU capacity (0..100), default: 100
                    Example: -q 25  => 25% of all cores; on 64 cores -> CPUQuota=1600%
  -c CORES_EQ       CPU quota as "equivalent cores" (float allowed), e.g., 2, 2.5

Other options:
  -A MASTER_ADDR    torchrun master addr (default: 10.10.0.2)
  -p MASTER_PORT    torchrun master port (default: 29500)
  -P NPROC          nproc-per-node (default: 1)
  -R REDIS_HOST     Redis host (default: 10.10.0.2)
  -E REDIS_PORT     Redis port (default: 6379)

Notes:
  * Any '--sudo_pass ...' detected in the trailing D2D args will be removed.
  * This script calls: 'sudo ufw disable' and 'bash kill_port.sh' before launch.
  * CPU limit applies only to torchrun and its children via systemd-run --scope.
EOF
  exit 1
}

IFACE=""; NNODES=""; NODE_RANK=""
QUOTA_PCT="100"      # 0..100 meaning % of total machine CPU
CORES_EQ=""          # if set, overrides QUOTA_PCT
MASTER_ADDR="10.10.0.2"
MASTER_PORT="29500"
NPROC=1
REDIS_HOST="10.10.0.2"
REDIS_PORT="6379"

# Parse flags until '--'
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) IFACE="$2"; shift 2 ;;
    -n) NNODES="$2"; shift 2 ;;
    -r) NODE_RANK="$2"; shift 2 ;;
    -q) QUOTA_PCT="$2"; shift 2 ;;   # numeric 0..100 (decimals ok)
    -c) CORES_EQ="$2"; shift 2 ;;    # decimal cores (e.g., 2.5)
    -A) MASTER_ADDR="$2"; shift 2 ;;
    -p) MASTER_PORT="$2"; shift 2 ;;
    -P) NPROC="$2"; shift 2 ;;
    -R) REDIS_HOST="$2"; shift 2 ;;
    -E) REDIS_PORT="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Remaining args are for D2Dexp_runner.py
D2D_ARGS=("$@")

# Validate requireds
[[ -z "$IFACE" || -z "$NNODES" || -z "$NODE_RANK" ]] && usage

# Host core count
TOTAL_CORES="$(nproc --all 2>/dev/null || echo 1)"
if ! [[ "$TOTAL_CORES" =~ ^[0-9]+$ ]] || [[ "$TOTAL_CORES" -le 0 ]]; then
  TOTAL_CORES=1
fi

# Compute effective CPUQuota (percent of a single CPU) for systemd-run
# If CORES_EQ given: CPUQuota = 100% * CORES_EQ
# Else: CPUQuota = (QUOTA_PCT% of TOTAL_CORES) = QUOTA_PCT * TOTAL_CORES %
CPU_QUOTA_EFFECTIVE="$(awk -v cores="${TOTAL_CORES}" -v pct="${QUOTA_PCT}" -v ce="${CORES_EQ:-}" 'BEGIN{
  # sanitize inputs
  if (cores <= 0) cores = 1;
  # if cores-eq provided, it wins
  if (ce != "") {
    q = 100.0 * ce;              # e.g., 2.5 cores -> 250%
  } else {
    # clamp pct to [0,100]
    p = pct + 0.0;
    if (p < 0) p = 0;
    if (p > 100) p = 100;
    q = (p/100.0) * cores * 100; # e.g., 25% of 64 cores -> 1600%
  }
  if (q < 1) q = 1;              # avoid 0% starvation
  printf("%.2f%%", q);
}')"

# Strip any --sudo_pass and its value from D2D_ARGS
FILTERED_ARGS=()
skip_next=0
for ((i=0; i<${#D2D_ARGS[@]}; i++)); do
  if [[ $skip_next -eq 1 ]]; then
    skip_next=0; continue
  fi
  if [[ "${D2D_ARGS[$i]}" == "--sudo_pass" ]]; then
    skip_next=1; continue
  fi
  if [[ "${D2D_ARGS[$i]}" == --sudo_pass=* ]]; then
    continue
  fi
  FILTERED_ARGS+=("${D2D_ARGS[$i]}")
done

echo "== Prep steps (need sudo) =="
sudo ufw disable || true
if [[ -f kill_port.sh ]]; then
  bash kill_port.sh || true
fi

echo "== Launching torchrun with CPU limit =="
echo "Host cores (nproc --all): ${TOTAL_CORES}"
if [[ -n "${CORES_EQ}" ]]; then
  echo "Requested cores: ${CORES_EQ}  ->  CPUQuota=${CPU_QUOTA_EFFECTIVE}"
else
  echo "Requested machine share: ${QUOTA_PCT}% of ${TOTAL_CORES} cores  ->  CPUQuota=${CPU_QUOTA_EFFECTIVE}"
fi
echo "IFACE=${IFACE}  NNODES=${NNODES}  NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}  NPROC=${NPROC}"
echo "REDIS=${REDIS_HOST}:${REDIS_PORT}"
echo "D2D args: ${FILTERED_ARGS[*]}"

# Run torchrun in a systemd scope with CPU quota
systemd-run --scope --quiet -p "CPUQuota=${CPU_QUOTA_EFFECTIVE}" -- \
  env PP_BW_IF="${IFACE}" \
      NCCL_SOCKET_IFNAME="${IFACE}" \
      GLOO_SOCKET_IFNAME="${IFACE}" \
      NCCL_IB_DISABLE=1 \
      REDIS_HOST="${REDIS_HOST}" \
      REDIS_PORT="${REDIS_PORT}" \
  torchrun \
    --nproc-per-node="${NPROC}" \
    --nnodes="${NNODES}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    D2Dexp_runner.py "${FILTERED_ARGS[@]}"
