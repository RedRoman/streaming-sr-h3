#!/usr/bin/env bash
# Publish OUR OWN trained checkpoints as a GitHub release. Needs a PAT in /cache/zihaohe/.gh_token with
# contents:write -- the SSH key authenticates `git push` but the REST API does not accept it, and
# gh is not installed on this host.
#
# Why a release and not the git history: weights over 100 MB cannot be committed without git-lfs (not
# installed, and GitHub's free LFS quota is 1 GB), and .gitignore records the standing policy that
# datasets and third-party weights are not ours to redistribute.
#
# What is NOT uploaded, deliberately:
#   baselines/openh3 (20 GB), baselines/omnibench (3.2 GB), LTX-2 (46 GB), RealVSR (7.4 GB)
#       third-party, under their own licenses.
#   assets/openh3_split/*.safetensors (15 GB)
#       rank-axis slices and relayouts of MiniMax-H3-Community-licensed weights, so the same question
#       applies. They are byte-reproducible from split_openh3_lora.py + convert_openh3_to_comfy.py,
#       whose correctness is pinned by results/deploy/openh3_convert_verify.json (max relative error 0.0
#       against Kijai's independently converted file).
#   results/lora_c_r128_p0/lora_step002000.safetensors (2.39 GB)
#       over GitHub's 2 GB per-asset limit, and it is a negative result (rank 128 lost to rank 32 on
#       null cosine by 0.0155 on 86 of 96 clips). Reproducible from code/lora_arms.sh.
set -u
R=/cache/zihaohe/streaming_sr0
REPO=RedRoman/streaming-sr-h3
TAG=${1:-ckpt-20261110}
T=$(cat /cache/zihaohe/.gh_token 2>/dev/null | tr -d '\r\n')
[ -n "$T" ] || { echo 'no token at /cache/zihaohe/.gh_token' >&2; exit 1; }
api () { curl -sS -H "Authorization: Bearer $T" -H 'Accept: application/vnd.github+json' "$@"; }

code=$(api -o /tmp/repo.json -w '%{http_code}' "https://api.github.com/repos/$REPO")
[ "$code" = 200 ] || { echo "token cannot read $REPO (http $code)" >&2; exit 1; }

ID=$(api "https://api.github.com/repos/$REPO/releases/tags/$TAG" | "$R/venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null)
if [ -z "$ID" ]; then
  ID=$(api -X POST "https://api.github.com/repos/$REPO/releases" \
    -d "{\"tag_name\":\"$TAG\",\"target_commitish\":\"archive/modelarts-20261103\",\"name\":\"checkpoints $TAG\",\"body\":\"Our own trained checkpoints for the commit on archive/modelarts-20261103. Third-party and derived-from-third-party weights are excluded; see code/push_weights.sh for what and why.\",\"draft\":false,\"prerelease\":false}" \
    | "$R/venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
fi
[ -n "$ID" ] || { echo 'could not create or find the release' >&2; exit 1; }
echo "release id $ID tag $TAG"

up () {  # up <path> <asset name>
  local p=$1 n=$2
  [ -f "$p" ] || { echo "missing $p"; return; }
  local have
  have=$(api "https://api.github.com/repos/$REPO/releases/$ID/assets" \
         | "$R/venv/bin/python" -c "import json,sys; print(any(a['name']=='$n' for a in json.load(sys.stdin)))")
  [ "$have" = True ] && { echo "skip $n (already uploaded)"; return; }
  echo "uploading $n ($(stat -c%s "$p") bytes)"
  curl -sS -X POST -H "Authorization: Bearer $T" -H 'Content-Type: application/octet-stream' \
    --data-binary @"$p" \
    "https://uploads.github.com/repos/$REPO/releases/$ID/assets?name=$n" \
    -o /tmp/up.json -w "  http=%{http_code} %{size_upload} bytes\n"
}

up "$R/results/stage1_lora/lora_step002000.safetensors"      stage1_lora_step002000_r32.safetensors
up "$R/results/range_branch/r5_k3.pt"                        range_branch_r5_k3.pt
up "$R/results/lora_a_r32_p0/lora_step002000.safetensors"    lora_a_r32_perc0_step002000.safetensors
up "$R/results/lora_b_r32_p4/lora_step002000.safetensors"    lora_b_r32_perc4_step002000.safetensors
up "$R/results/deploy/m0_W.pt"                               m0_latent_map_W.pt
api "https://api.github.com/repos/$REPO/releases/$ID/assets" \
  | "$R/venv/bin/python" -c 'import json,sys; [print("  %-52s %10d" % (a["name"], a["size"])) for a in json.load(sys.stdin)]'
