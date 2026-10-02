#!/usr/bin/env python3
"""Extract an Alignment-format projector.pt from an Agentic RL checkpoint.

Pass --agentic_rl-ckpt for a global_step_N/actor directory, --from-alignment
for the Alignment checkpoint used to initialize that run, and --out for the
destination projector.pt. This exports a projector, not a complete trained policy.

Merge only dyad_residual_head.projector.* tensors. Encoder hidden-state and mask
buffers describe the run's action set and must not enter the exported weights.
Copy model, projector, and scale metadata from the original Alignment checkpoint.

Reconstruct DTensor shards from their placements in rank order and validate the
global shape. Use local tensors rather than collectives: checkpoint reading has
no process group. Replicated tensors must agree across ranks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
PROJECT = SELF.parents[5]                      # Dynamic_ExpA
sys.path.insert(0, str(PROJECT))

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

PREFIX = "dyad_residual_head."

# Export projector parameters only; encoder buffers belong to the source run's action set.
WANTED = PREFIX + "projector."

# Copy metadata from the Alignment checkpoint that initialized this run.
CARRIED = ("projector", "scale", "policy_model", "encoder_model",
           "policy_hidden", "encoder_hidden")


def load_shards(ckpt_dir: Path) -> list[dict]:
    shards = sorted(ckpt_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise SystemExit(
            f"[extract] {ckpt_dir} 下没有 model_world_size_*_rank_*.pt。\n"
            f"  要指到一个具体的 global_step_N/actor/ 目录，不是 run 根目录。\n"
            f"  这个目录下有：{sorted(p.name for p in ckpt_dir.iterdir())[:8]}")
    import torch
    print(f"[extract] {len(shards)} 个分片")
    return [torch.load(p, map_location="cpu", weights_only=False) for p in shards]


def unwrap(shard: dict) -> dict:
    """Accept either a direct state_dict or a checkpoint wrapper containing it."""
    for key in ("model", "state_dict"):
        if key in shard and isinstance(shard[key], dict):
            return shard[key]
    return shard


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agentic_rl-ckpt", required=True,
                    help="global_step_N/actor/ 目录（含 model_world_size_*_rank_*.pt）")
    ap.add_argument("--from-alignment", required=True,
                    help="这个 Agentic RL run 当初 init 用的那份 Alignment run（七个字段从它抄）")
    ap.add_argument("--out", required=True, help="写到哪，一般是 <agentic_rl run>/projector.pt")
    args = ap.parse_args()

    import torch
    from agent_system.policies.dyad.models.action_head_factory import _resolve_projector_init

    ckpt_dir = Path(args.agentic_rl_ckpt)
    if ckpt_dir.name != "actor" and (ckpt_dir / "actor").is_dir():
        ckpt_dir = ckpt_dir / "actor"

    base_path = _resolve_projector_init(args.from_alignment)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    print(f"[extract] 元数据来自 {base_path}")

    shards = [unwrap(s) for s in load_shards(ckpt_dir)]

    keys = [k for k in shards[0] if k.startswith(WANTED)]
    if not keys:
        sample = [k for k in list(shards[0])[:6]]
        raise SystemExit(
            f"[extract] {RED}分片里没有 {WANTED}* 的键{RESET}\n"
            f"  这个 checkpoint 是在 DYAD_ENCODER_ENABLED=1 下训的吗？"
            f"（encoder 关掉时不存在 residual head）\n"
            f"  前几个键长这样：{sample}")

    def local_of(t):
        """Return a DTensor local shard or an ordinary tensor unchanged.

        Avoid full_tensor(), which requires a process group and collective communication.
        """
        loc = getattr(t, "_local_tensor", None)
        return loc if loc is not None else t

    def shard_dim(t):
        for pl in getattr(t, "placements", ()) or ():
            if getattr(pl, "is_shard", lambda: False)():
                return pl.dim
        return None

    state: dict = {}
    for key in keys:
        parts = [s[key] for s in shards]
        short = key[len(PREFIX):]
        dim = shard_dim(parts[0])
        if dim is None:
            # Replicated tensors must agree across ranks.
            locs = [local_of(t) for t in parts]
            if not all(torch.equal(locs[0], t) for t in locs[1:]):
                raise SystemExit(
                    f"[extract] {RED}{short} 没有分片，但各 rank 的值不同{RESET}\n"
                    f"  取任何一份都是在猜。这说明 checkpoint 的写法与这里的假设不一致。")
            state[short] = locs[0].clone()
            print(f"  {short}: 未分片，各 rank 相同 {tuple(locs[0].shape)}")
        else:
            # Concatenate shards in rank order along their declared dimension, then check the global shape.
            want = tuple(parts[0].shape)
            merged = torch.cat([local_of(t) for t in parts], dim=dim)
            if tuple(merged.shape) != want:
                raise SystemExit(
                    f"[extract] {RED}{short} 拼接后 {tuple(merged.shape)} != DTensor 自报的全局形状 {want}{RESET}\n"
                    f"  分片维={dim}，各 rank 本地形状 {[tuple(local_of(t).shape) for t in parts]}。\n"
                    f"  分片规则与这里的假设不一致，拼错的 projector 照样加载得进去。")
            state[short] = merged.contiguous()
            print(f"  {short}: Shard(dim={dim}) {len(parts)} 片拼回 {tuple(merged.shape)}")

    # Validate each reconstructed shape against the original Alignment projector.
    ref = base["state_dict"]
    bad = [k for k in ref if k not in state or tuple(state[k].shape) != tuple(ref[k].shape)]
    if bad:
        raise SystemExit(
            f"[extract] {RED}抽出来的张量与 Alignment 那份形状对不上：{bad}{RESET}\n"
            f"  Alignment: {{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in ref.items())}}}\n"
            f"  抽出来:  {{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in state.items())}}}\n"
            f"  多半是分片拼接的方式不对 —— 见本文件抬头「分片」那一段。")
    extra = sorted(set(state) - set(ref))
    if extra:
        raise SystemExit(
            f"[extract] {RED}多出 Alignment 没有的键：{extra}{RESET}\n"
            f"  两个 encoder buffer 出现在这里，说明 checkpoint 是在 set_encoder_cache 之后写的，"
            f"它带着那次 run 的动作集 —— 装到别的 schema 上不会有任何形状报错。")

    payload = {"state_dict": state}
    payload.update({k: base[k] for k in CARRIED})
    payload["source_agentic_rl_ckpt"] = str(ckpt_dir)
    payload["source_alignment"] = str(base_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[extract] {GREEN}写好{RESET} {out}")
    print(f"  下一步：DYAD_ENCODER_PROJECTOR_INIT={out.parent}")
    print("  核对一下：python experiments/shared/train_eval/scripts/prepare.py projector \\")
    print(f"      --projector-init {out.parent} --model-name <MODEL_NAME>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
