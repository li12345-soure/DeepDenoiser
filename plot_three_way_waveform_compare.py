import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plot teacher/base/distill three-way waveform comparison')
    parser.add_argument('--teacher_npz', default=r'.\result_teacher_float_eval_same_sample.npz')
    parser.add_argument('--base_npz', default=r'.\result_A_d4_r6_base_float_tf_same_sample.npz')
    parser.add_argument('--distill_npz', default=r'.\result_A_d4_r6_distill_v1_float_tf_same_sample.npz')
    parser.add_argument('--save_png', default=r'.\three_way_waveform_compare.png')
    parser.add_argument('--max_points', type=int, default=1200, help='max time points to draw from the start')
    args = parser.parse_args()

    teacher = np.load(args.teacher_npz)['denoised_waveform']
    base = np.load(args.base_npz)['denoised_waveform']
    distill = np.load(args.distill_npz)['denoised_waveform']

    n = min(args.max_points, teacher.shape[0])
    x = np.arange(n)

    fig = plt.figure(figsize=(14, 10))

    for ch in range(min(3, teacher.shape[1])):
        ax = fig.add_subplot(4, 1, ch + 1)
        ax.plot(x, teacher[:n, ch], label=f'Teacher ch{ch}')
        ax.plot(x, base[:n, ch], label=f'Base A ch{ch}')
        ax.plot(x, distill[:n, ch], label=f'Distill A ch{ch}')
        ax.set_title(f'Denoised waveform channel {ch}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    ax4 = fig.add_subplot(4, 1, 4)
    diff_tb = np.abs(teacher[:n, 0] - base[:n, 0])
    diff_td = np.abs(teacher[:n, 0] - distill[:n, 0])
    ax4.plot(x, diff_tb, label='|teacher-base| ch0')
    ax4.plot(x, diff_td, label='|teacher-distill| ch0')
    ax4.set_title('Absolute waveform difference (channel 0)')
    ax4.set_xlabel('Time index')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')

    def wave_metrics(a, b):
        mae = float(np.mean(np.abs(a - b)))
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        cosine = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return mae, rmse, cosine

    tb_mae, tb_rmse, tb_cos = wave_metrics(teacher, base)
    td_mae, td_rmse, td_cos = wave_metrics(teacher, distill)
    bd_mae, bd_rmse, bd_cos = wave_metrics(base, distill)

    summary = (
        f'teacher vs base: mae={tb_mae:.4f}, rmse={tb_rmse:.4f}, cosine={tb_cos:.6f}\n'
        f'teacher vs distill: mae={td_mae:.4f}, rmse={td_rmse:.4f}, cosine={td_cos:.6f}\n'
        f'base vs distill: mae={bd_mae:.4f}, rmse={bd_rmse:.4f}, cosine={bd_cos:.6f}'
    )
    fig.text(0.02, 0.01, summary, fontsize=10, family='monospace')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(args.save_png, dpi=150, bbox_inches='tight')
    print(f'[OK] saved: {args.save_png}')


if __name__ == '__main__':
    main()
