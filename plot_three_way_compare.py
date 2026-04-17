import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plot teacher/base/distill three-way logits-preds comparison')
    parser.add_argument('--teacher_npz', default=r'.\result_teacher_float_eval_same_sample.npz')
    parser.add_argument('--base_npz', default=r'.\result_A_d4_r6_base_float_tf_same_sample.npz')
    parser.add_argument('--distill_npz', default=r'.\result_A_d4_r6_distill_v1_float_tf_same_sample.npz')
    parser.add_argument('--save_png', default=r'.\three_way_logits_compare.png')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--row', type=int, default=15)
    args = parser.parse_args()

    teacher = np.load(args.teacher_npz)
    base = np.load(args.base_npz)
    distill = np.load(args.distill_npz)

    t_logits = teacher['logits']
    b_logits = base['logits']
    d_logits = distill['logits']

    t_preds = teacher['preds']
    b_preds = base['preds']
    d_preds = distill['preds']

    x = np.arange(t_logits.shape[2])
    s = args.sample
    r = args.row

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(x, t_logits[s, r, :, 0], label='Teacher logits ch0')
    ax1.plot(x, b_logits[s, r, :, 0], label='Base A logits ch0')
    ax1.plot(x, d_logits[s, r, :, 0], label='Distill A logits ch0')
    ax1.set_title(f'Logits channel 0 | sample={s}, row={r}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(x, t_logits[s, r, :, 1], label='Teacher logits ch1')
    ax2.plot(x, b_logits[s, r, :, 1], label='Base A logits ch1')
    ax2.plot(x, d_logits[s, r, :, 1], label='Distill A logits ch1')
    ax2.set_title('Logits channel 1')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(x, t_preds[s, r, :, 0], label='Teacher prob ch0')
    ax3.plot(x, b_preds[s, r, :, 0], label='Base A prob ch0')
    ax3.plot(x, d_preds[s, r, :, 0], label='Distill A prob ch0')
    ax3.set_title('Softmax prob channel 0')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(x, np.abs(t_preds[s, r, :, 0] - b_preds[s, r, :, 0]), label='|teacher-base| prob ch0')
    ax4.plot(x, np.abs(t_preds[s, r, :, 0] - d_preds[s, r, :, 0]), label='|teacher-distill| prob ch0')
    ax4.set_title('Absolute probability difference')
    ax4.set_xlabel('Feature width index')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(args.save_png, dpi=150, bbox_inches='tight')
    print(f'[OK] saved: {args.save_png}')


if __name__ == '__main__':
    main()
