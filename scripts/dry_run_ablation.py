"""
三组实验（A0 / A1 / C-on-A1）端到端演练。

**刻意用子进程逐步调用 `scripts/train_dual.py`**，与 `scripts/run_ablation.sh`
完全一致。这样演练同时验证了：
  · 真实 CLI 路径（--config / --mode / --prefix / --static_init）能用
  · A0 / A1 / C 三条训练路径都能跑通
  · C 能从 A1 的静态检查点正确热启动（变体一致，不报警）
  · C 若误用 A0 的检查点，变体护栏会报警（反向验证）
  · 每个实验是独立进程，内存逐步释放（与云端行为一致）
  · 分析脚本在「没有纯动态模型」时自动改为 A1 vs C 的误差分解
  · A1 vs A0 的配对检验路径可用

用极小子集 + 1 epoch，只验证流程，不验证性能。
真实训练请用 `bash scripts/run_ablation.sh`（完整数据、完整轮数）。

用法:
    python scripts/dry_run_ablation.py
    python scripts/dry_run_ablation.py --writers 1 --threads 2
"""

from __future__ import annotations

import os
import sys
import argparse
import shutil
import subprocess

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

import pandas as pd                                       # noqa: E402


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
        return ok

    def summary(self):
        print("\n" + "=" * 74)
        print(f"结果: {self.passed} 项通过, {self.failed} 项失败")
        print("=" * 74)
        return self.failed == 0


def make_subset(src, out, n_writers, train_pages, test_pages):
    df = pd.read_csv(src, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    writers = sorted(df['writer_id'].unique())[:n_writers]
    parts = []
    for split, per in (('Train', train_pages), ('Test', test_pages)):
        sub = df[(df['split'] == split) & (df['writer_id'].isin(writers))]
        parts.append(sub.groupby('writer_id', group_keys=False).head(per))
    out_df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    out_df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"   子集 metadata: {out} "
          f"(writers={len(writers)}, Train={int((out_df['split']=='Train').sum())} 页, "
          f"Test={int((out_df['split']=='Test').sum())} 页)")
    return out


def write_dry_config(path, metadata, work_dir, threads):
    """生成一份极小配置，结构与 configs/dual_stream.yaml 一致。

    关键点：stages_by_mode 必须按模式分开——纯静态用单阶段，
    这样演练才会覆盖「static 不会误用双路三阶段」这条修复。
    """
    cfg = {
        'mode': 'dual',
        'model': {
            'feature_dim': 512,
            'in_chans': 64,
            'pretrained': False,
            'pretrained_path': None,
            'drop_path': 0.0,
            'dropout_head': 0.1,
            'dropout_classifier': 0.0,
            'dynamic_dim': 256,
            'seq_len': 200,
            'dynamic_in_dim': 2,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 128,
            'dynamic_dropout': 0.0,
            'pooling': 'attention',
            'pos_encoding': 'sinusoidal',
            'fusion_hidden': 128,
            'fusion_dropout': 0.1,
            'freeze_static': True,
            'static_init_path': None,
        },
        'data': {
            'batch_size': 8,
            'num_workers': 0,
            'page_workers': 0,
            'segment_workers': 0,
            'seq_len': 200,
            'dynamic_speed_clip': 10.0,
        },
        'training': {
            'weight_decay': 0.05,
            'warmup_epochs': 0,
            'eval_frequency': 1,
            'patience': 99,
            'grad_clip': 5.0,
            'apply_drop': True,
            'resume': False,
            'lr_base': 0.000667,
            'stages_by_mode': {
                # 刻意镜像 configs/dual_stream.yaml 的结构：
                # 纯静态 = 单阶段；双路 = 三阶段（含解冻）。
                # 这样演练才能真正验证「static 不会误用双路三阶段」这条修复。
                'static': [
                    {'name': 'static_baseline', 'epochs': 1, 'lr': 1.0,
                     'freeze_static': False, 'static_lr_scale': 1.0},
                ],
                'dynamic': [
                    {'name': 'stage1_warmup', 'epochs': 1, 'lr': 1e-3,
                     'freeze_static': True, 'static_lr_scale': 0.0},
                    {'name': 'stage2_converge', 'epochs': 1, 'lr': 5e-4,
                     'freeze_static': True, 'static_lr_scale': 0.0},
                    {'name': 'stage3_finetune', 'epochs': 1, 'lr': 1e-4,
                     'freeze_static': True, 'static_lr_scale': 0.0},
                ],
                'dual': [
                    {'name': 'stage1_head_warmup', 'epochs': 1, 'lr': 1e-3,
                     'freeze_static': True, 'static_lr_scale': 0.0},
                    {'name': 'stage2_head_converge', 'epochs': 1, 'lr': 5e-4,
                     'freeze_static': True, 'static_lr_scale': 0.0},
                    {'name': 'stage3_joint_finetune', 'epochs': 1, 'lr': 1e-4,
                     'freeze_static': False, 'static_lr_scale': 0.1},
                ],
            },
        },
        'checkpoint': {'save_dir': os.path.join(work_dir, 'checkpoints'),
                       'prefix': 'dry'},
        'logging': {'log_dir': os.path.join(work_dir, 'logs'),
                    'log_file': 'dry.csv'},
        'metadata_file': metadata,
        'seed': 0,
        '_threads': threads,
    }
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def run(cmd, label, log_path):
    """跑一个子进程，输出落盘；返回 (returncode, output)。"""
    print(f"\n$ {' '.join(cmd)}")
    with open(log_path, 'w', encoding='utf-8') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        out = f.read()
    tail = "\n".join(out.splitlines()[-25:])
    print(f"  [{label}] returncode={proc.returncode}  日志: {log_path}")
    if proc.returncode != 0:
        print("  --- 输出末尾 ---")
        print("\n".join(f"  {l}" for l in tail.splitlines()))
    return proc.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', type=str,
                    default='data/features/metadata_small.csv')
    ap.add_argument('--writers', type=int, default=1)
    ap.add_argument('--train_pages', type=int, default=1)
    ap.add_argument('--test_pages', type=int, default=1)
    ap.add_argument('--work_dir', type=str, default='_dry_out')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()

    if os.path.isdir(args.work_dir):
        shutil.rmtree(args.work_dir, ignore_errors=True)
    os.makedirs(args.work_dir, exist_ok=True)
    logs = os.path.join(args.work_dir, 'logs')
    os.makedirs(logs, exist_ok=True)

    meta = make_subset(args.metadata,
                       os.path.join(args.work_dir, 'metadata_dry.csv'),
                       args.writers, args.train_pages, args.test_pages)
    cfg = write_dry_config(os.path.join(args.work_dir, 'dry_config.yaml'),
                           meta, args.work_dir, args.threads)

    env = dict(os.environ, PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    py = sys.executable
    pycmd = [py, '-u']          # -u：不缓冲，被环境硬杀时日志里仍有进度
    ckpt = os.path.join(args.work_dir, 'checkpoints')

    rep = Report()

    # ---------------- 变体护栏的语境区分（不需训练，先验证） ----------------
    print("\n" + "=" * 74)
    print("步骤 0/5: 变体护栏的语境区分（static 提示 / dual 警告）")
    print("=" * 74)
    import io
    import contextlib
    from src.models.dual_stream import assert_variant_compatible

    base_ckpt = os.path.join('outputs', 'checkpoints', 'convnext_best_9617.pth')
    if os.path.exists(base_ckpt):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert_variant_compatible(base_ckpt, True, True, mode='static')
        txt = buf.getvalue()
        rep.check("★ static 模式热启动 A1：给提示而非报错",
                  '推荐做法' in txt and '不匹配' not in txt)
        rep.check("  提示里说明了方法学影响（继续训练的混淆）",
                  '继续训练' in txt and '对照' in txt)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert_variant_compatible(base_ckpt, True, True, mode='dual')
        txt = buf.getvalue()
        rep.check("★ dual 模式用 A0 检查点：强警告",
                  '不匹配' in txt and '冻结' in txt)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert_variant_compatible(base_ckpt, False, False, mode='static')
        rep.check("A0 检查点 + A0 运行：正常通过",
                  '语义一致' in buf.getvalue())
    else:
        print(f"  [SKIP] 找不到 {base_ckpt}，跳过护栏语境检查")

    # ---------------- A1（热启动微调） ----------------
    print("\n" + "=" * 74)
    print("步骤 1/5: 实验 A1（纯静态 + 修正流水线，从 A0 检查点热启动）")
    print("=" * 74)
    a1_args = pycmd + ['scripts/train_dual.py', '--config', cfg,
                       '--mode', 'static', '--prefix', 'static_a1',
                       '--static_epochs', '1', '--static_lr', '0.4']
    if os.path.exists(base_ckpt):
        a1_args += ['--static_init', base_ckpt]
    else:
        a1_args += ['--static_init', 'none']
    rc1, out1 = run(a1_args, 'A1', os.path.join(logs, 'a1.log'))
    a1_best = os.path.join(ckpt, 'static_a1_best.pth')
    rep.check("A1 子进程退出码为 0", rc1 == 0, f"rc={rc1}")
    rep.check("A1 检查点已落盘", os.path.exists(a1_best), a1_best)
    rep.check("A1 打印了特征版本",
              'v2.1' in out1 and '1 位图 + 63 签名' in out1)
    rep.check("A1 解析为单阶段 static_baseline（未误用双路三阶段）",
              'static_baseline' in out1 and 'stage1_head_warmup' not in out1)
    rep.check("--static_epochs 生效（1 轮）",
              'epochs=1' in out1)
    if os.path.exists(base_ckpt):
        rep.check("A1 从 A0 检查点热启动并给出提示（非报错）",
                  '推荐做法' in out1)
        rep.check("A1 成功加载骨干网权重",
                  '已加载' in out1 and '个张量' in out1)

    # ---------------- A0 对照（可选） ----------------
    print("\n" + "=" * 74)
    print("步骤 2/5: 可选对照 A0ctrl（A0 特征 + 与 A1 相同的起点/配方）")
    print("=" * 74)
    a0c_best = os.path.join(ckpt, 'static_a0ctrl_best.pth')
    if os.path.exists(base_ckpt):
        a0c_args = pycmd + ['scripts/train_dual.py', '--config', cfg,
                            '--mode', 'static', '--prefix', 'static_a0ctrl',
                            '--static_init', base_ckpt,
                            '--static_epochs', '1', '--static_lr', '0.4']
        rc0c, out0c = run(a0c_args, 'A0ctrl', os.path.join(logs, 'a0ctrl.log'))
        rep.check("A0ctrl 子进程退出码为 0", rc0c == 0, f"rc={rc0c}")
        rep.check("A0ctrl 检查点已落盘", os.path.exists(a0c_best), a0c_best)
    else:
        print("  [SKIP] 没有基准检查点，跳过 A0 对照")

    # ---------------- C on A1 ----------------
    print("\n" + "=" * 74)
    print("步骤 3/5: 实验 C（双路融合，建在 A1 之上）")
    print("=" * 74)
    rc2, out2 = run(pycmd + ['scripts/train_dual.py', '--config', cfg,
                             '--mode', 'dual', '--prefix', 'dual_a1',
                             '--static_init', a1_best],
                    'C', os.path.join(logs, 'c.log'))
    c_best = os.path.join(ckpt, 'dual_a1_best.pth')
    rep.check("C 子进程退出码为 0", rc2 == 0, f"rc={rc2}")
    rep.check("C 检查点已落盘", os.path.exists(c_best), c_best)
    rep.check("C 从 A1 热启动且变体一致",
              '变体一致' in out2, "（不应出现任何警告）")
    rep.check("C 解析为双路三阶段",
              'stage1_head_warmup' in out2 and 'stage3_joint_finetune' in out2)
    rep.check("C 阶段3 解冻静态分支",
              'static_lr_scale=0.1' in out2)

    # ---------------- 分析 ----------------
    print("\n" + "=" * 74)
    print("步骤 4/5: 互补性分析（无纯动态模型 -> 应自动改为 A1 vs C）")
    print("=" * 74)
    if os.path.exists(a1_best) and os.path.exists(c_best):
        cmd = pycmd + ['scripts/analyze_complementarity.py',
                       '--metadata', meta,
                       '--static_ckpt', a1_best,
                       '--dual_ckpt', c_best,
                       '--dynamic_ckpt', os.path.join(ckpt, 'nonexistent.pth'),
                       '--out_dir', os.path.join(args.work_dir, 'analysis'),
                       '--batch_size', '8', '--num_workers', '0',
                       '--page_workers', '0', '--segment_workers', '0',
                       '--n_test', '2', '--tsne_samples', '100']
        if os.path.exists(a0c_best):
            cmd += ['--baseline_ckpt', a0c_best]
        rca, outa = run(cmd, 'analyze',
                        os.path.join(logs, 'analysis.log'))
        rep.check("分析脚本退出码为 0", rca == 0, f"rc={rca}")
        rep.check("误差分解自动改为 static vs dual",
                  'static vs dual' in outa)
        rep.check("日志里出现 A1 vs A0 的检验", 'A1 vs A0' in outa)
        sig_csv = os.path.join(args.work_dir, 'analysis', 'significance.csv')
        if os.path.exists(sig_csv):
            with open(sig_csv, encoding='utf-8-sig') as f:
                sig_txt = f.read()
            rep.check("significance.csv 含 static_a1_vs_a0 行",
                      'static_a1_vs_a0' in sig_txt)
            rep.check("significance.csv 含 dual_vs_static 行",
                      'dual_vs_static' in sig_txt)
        else:
            rep.check("significance.csv 存在", False)
        rep.check("分析脚本按检查点重建了架构（不依赖脚本默认值）",
                  '架构(取自检查点)' in outa)
        for f in ('summary.md', 'error_quadrants.png',
                  'tsne_static_dynamic.png', 'paired_20test_records.csv',
                  'cca_correlations.csv'):
            p = os.path.join(args.work_dir, 'analysis', f)
            rep.check(f"产出 {f}", os.path.exists(p))

    ok = rep.summary()
    print(f"\n演练产物目录: {os.path.abspath(args.work_dir)}")
    print("（这是流程演练，不是性能结果；真实训练请用 bash scripts/run_ablation.sh）")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
