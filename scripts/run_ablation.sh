#!/usr/bin/env bash
# ============================================================================
# 伪时序双路融合 —— 实验流程（云端 Linux）
#
# 默认流程（A0 不重跑，直接用已发表的 96.17%）:
#
#   a1      : 纯静态 + 修正流水线，从 convnext_best_9617.pth **热启动微调**
#             -> 这是关键路径的瓶颈，热启动让它比从零训练快数倍
#   c       : 双路融合，静态分支从 A1 的检查点热启动
#             -> 必须先跑完 a1
#   analyze : CCA + 误差四象限 + 配对 t 检验 + t-SNE
#
# 可选:
#
#   a0ctrl  : A0 特征 + 与 A1 完全相同的热启动/轮数/配方
#             -> 干净归因用。开销与 a1 相同（**不是**从零重跑 A0），
#                用来把 A1-vs-96.17% 拆成「特征修复」与「继续训练」两份。
#   a1bitmap: 只开 separator_aware，把 A1 的提升再拆成抬笔分段 / 签名布局两份。
#
# 用法:
#   bash scripts/run_ablation.sh                     # a1 -> c -> analyze
#   bash scripts/run_ablation.sh a1 a0ctrl c analyze # 含干净归因对照
#   bash scripts/run_ablation.sh a1                  # 只跑 A1
#   SKIP_VERIFY=1 bash scripts/run_ablation.sh       # 跳过前置验证
#
# 中断后直接重跑同一条命令即可：断点续训会精确恢复到「阶段 k 的第 n 轮」
# （含 AdamW 动量与调度器状态）。
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# ---- 可调参数 --------------------------------------------------------------
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/dual_stream.yaml}"
METADATA="${METADATA:-data/features/metadata.csv}"
CKPT_DIR="${CKPT_DIR:-outputs/checkpoints}"
N_TEST="${N_TEST:-20}"            # 互补性分析里的 20-test 轮数，0 表示跳过

# A1 / C 的静态流水线开关
A1_FLAGS=(--set separator_aware=true fix_signature_layout=true)

# A1 的热启动来源：已发表的 96.17% 纯静态基线
BASE_CKPT="${BASE_CKPT:-$CKPT_DIR/convnext_best_9617.pth}"

# A1 微调的轮数与学习率。
# 从已收敛的检查点出发，不需要 50 轮；短余弦周期 + 较低 lr 即可。
# 留空则用 config 里的默认值（50 轮 / 5e-4），那是从零训练的配方。
A1_EPOCHS="${A1_EPOCHS:-15}"
A1_LR="${A1_LR:-0.4}"             # 占位值，实际 lr = 0.4 × lr_base × batch/256 ≈ 2e-4

# 各实验的 checkpoint 前缀
PREFIX_A0CTRL="static_a0ctrl"
PREFIX_A1="static_a1"
PREFIX_A1B="static_a1_bitmap"
PREFIX_C="dual_a1"

STEPS=("$@")
if [ "${#STEPS[@]}" -eq 0 ]; then
  STEPS=(a1 c analyze)
fi

a1_static_args() {
  # A1 系列（含 a0ctrl / a1bitmap）共用的热启动与轮数设置
  echo "--static_init $BASE_CKPT --static_epochs $A1_EPOCHS --static_lr $A1_LR"
}

run_step() {
  local step="$1"
  case "$step" in
    a1)
      echo ""
      echo "############################################################"
      echo "# 实验 A1：纯静态 + 修正流水线"
      echo "#   热启动: $BASE_CKPT"
      echo "#   轮数=$A1_EPOCHS  lr占位=$A1_LR (实际 ≈ 2e-4)"
      echo "#   注意：热启动是**推荐做法**。A0 与 A1 的第 0 通道（位图）"
      echo "#         逐位相同，而位图是唯一有空间分辨力的通道，"
      echo "#         所以这次热启动语义匹配度很高、收敛远快于从零训练。"
      echo "############################################################"
      if [ ! -f "$BASE_CKPT" ]; then
        echo "❌ 找不到热启动检查点: $BASE_CKPT"
        echo "   请先上传 convnext_best_9617.pth，或用 BASE_CKPT=... 指定，"
        echo "   或改用 --static_init none 从零训练（会慢很多）。"
        exit 1
      fi
      "$PYTHON" scripts/train_dual.py --config "$CONFIG" \
        --mode static --prefix "$PREFIX_A1" \
        --static_init "$BASE_CKPT" \
        --static_epochs "$A1_EPOCHS" --static_lr "$A1_LR" \
        "${A1_FLAGS[@]}"
      ;;

    a0ctrl)
      echo ""
      echo "############################################################"
      echo "# 可选对照：A0 特征 + 与 A1 完全相同的起点/轮数/配方"
      echo "#   唯一差别 = 特征流水线（A0 原样）"
      echo "#   用途：把「A1 vs 已发表 96.17%」拆成"
      echo "#         「特征修复」与「继续训练」两份贡献"
      echo "#   开销与 a1 相同（不是从零重跑 A0，那才是贵的那个）"
      echo "############################################################"
      "$PYTHON" scripts/train_dual.py --config "$CONFIG" \
        --mode static --prefix "$PREFIX_A0CTRL" \
        --static_init "$BASE_CKPT" \
        --static_epochs "$A1_EPOCHS" --static_lr "$A1_LR"
      ;;

    a1bitmap)
      echo ""
      echo "############################################################"
      echo "# 可选归因：只修抬笔分段（separator_aware），不动签名布局"
      echo "#   配合 A1 可把提升拆成「抬笔分段修复」与「签名布局修复」"
      echo "############################################################"
      "$PYTHON" scripts/train_dual.py --config "$CONFIG" \
        --mode static --prefix "$PREFIX_A1B" \
        --static_init "$BASE_CKPT" \
        --static_epochs "$A1_EPOCHS" --static_lr "$A1_LR" \
        --set separator_aware=true
      ;;

    c)
      echo ""
      echo "############################################################"
      echo "# 实验 C：双路融合（静态分支从 A1 热启动）"
      echo "#   参照基线是 A1 —— 这是更严格的对比"
      echo "############################################################"
      local a1_ckpt="$CKPT_DIR/${PREFIX_A1}_best.pth"
      if [ ! -f "$a1_ckpt" ]; then
        echo "❌ 找不到 A1 的检查点: $a1_ckpt"
        echo "   请先跑完实验 A1（bash scripts/run_ablation.sh a1）。"
        echo "   C 必须建在 A1 上：A0/A1 的通道语义不同，用 A0 的检查点"
        echo "   热启动会让被冻结的静态分支一直产出错误特征。"
        exit 1
      fi
      "$PYTHON" scripts/train_dual.py --config "$CONFIG" \
        --mode dual --prefix "$PREFIX_C" \
        --static_init "$a1_ckpt" \
        "${A1_FLAGS[@]}"
      ;;

    verify)
      echo ""
      echo "############################################################"
      echo "# 前置验证：动态特征正确性 + 全套件冒烟测试 + 静态流水线体检"
      echo "############################################################"
      "$PYTHON" scripts/verify_dynamic.py
      "$PYTHON" scripts/smoke_test_dual.py --skip_e2e
      "$PYTHON" scripts/diagnose_static_features.py
      ;;

    analyze)
      echo ""
      echo "############################################################"
      echo "# 互补性分析：CCA + 误差四象限 + 配对 t 检验 + t-SNE"
      echo "#   误差分解自动对比 A1 vs C ——"
      echo "#   「最好的纯静态模型都错了、而融合对了」是更硬的证据"
      echo "############################################################"
      local extra=()
      if [ -f "$CKPT_DIR/${PREFIX_A0CTRL}_best.pth" ]; then
        extra=(--baseline_ckpt "$CKPT_DIR/${PREFIX_A0CTRL}_best.pth")
        echo "   （检测到 A0 对照模型，将附带 A1 vs A0ctrl 的配对检验）"
      fi
      "$PYTHON" scripts/analyze_complementarity.py \
        --metadata "$METADATA" \
        --static_ckpt "$CKPT_DIR/${PREFIX_A1}_best.pth" \
        --dual_ckpt   "$CKPT_DIR/${PREFIX_C}_best.pth" \
        --dynamic_ckpt "$CKPT_DIR/dynamic_stream_best.pth" \
        ${extra[@]+"${extra[@]}"} \
        --n_test "$N_TEST"
      ;;

    *)
      echo "❌ 未知步骤: $step"
      echo "   可选: verify / a1 / a0ctrl / a1bitmap / c / analyze"
      exit 1
      ;;
  esac
}

# ---- 前置验证 --------------------------------------------------------------
# 除非 SKIP_VERIFY=1，或者本次只做 analyze，否则先跑一遍验证。
# 用 if 而不是 `[ ] && cmd`：后者在条件为假时返回非零，会被 set -e 中断。
if [ "${SKIP_VERIFY:-0}" != "1" ]; then
  only_analyze=1
  for s in "${STEPS[@]}"; do
    if [ "$s" != "analyze" ]; then
      only_analyze=0
    fi
  done
  if [ "$only_analyze" -eq 0 ]; then
    run_step verify
  fi
fi

for step in "${STEPS[@]}"; do
  if [ "$step" = "verify" ]; then
    continue          # 已在前置块跑过，避免重复
  fi
  run_step "$step"
done

echo ""
echo "============================================================"
echo "🏁 全部完成"
echo "   checkpoint : $CKPT_DIR"
echo "   日志       : outputs/logs"
echo "   分析产出   : outputs/analysis"
echo "============================================================"
echo ""
echo "论文里的对照关系："
echo "  A0 已发表 96.17%        -> 不重跑，直接引用"
echo "  A1 ${PREFIX_A1}           -> 修正流水线后的纯静态基线"
echo "  C  ${PREFIX_C}            -> C vs A1 = 动态分支的增量价值"
if [ -f "$CKPT_DIR/${PREFIX_A0CTRL}_best.pth" ]; then
  echo "  A0ctrl ${PREFIX_A0CTRL}       -> A1 vs A0ctrl = 拆掉「继续训练」的贡献"
else
  echo ""
  echo "  ℹ️ 未跑 A0 对照。A1 相对已发表 96.17% 的差异里同时含"
  echo "     「特征修复」与「继续训练」两个因素。若要干净归因："
  echo "       bash scripts/run_ablation.sh a0ctrl"
  echo "     （开销与 A1 相同，远小于从零重跑 A0）"
fi
