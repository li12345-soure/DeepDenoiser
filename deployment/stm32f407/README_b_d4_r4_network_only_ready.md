# STM32F407 B_d4_r4 Network-only Ready

## Model

- Candidate: B_d4_r4 int8 logits
- TFLite: b_d4_r4_int8_logits.tflite
- Target MCU: STM32F407ZGT6
- X-CUBE-AI application mode: ApplicationTemplate

## X-CUBE-AI Analyze

- weights: 21,992 B
- activations: 102,132 B
- MACC: 14,712,726
- estimated total RAM: about 112 KB

## X-CUBE-AI Validate

- X-cross rmse: 0
- X-cross mae: 0
- X-cross cos: 1.0

## Application Build

- Project: b_d4_r4_app
- Build result: 0 errors, 0 warnings
- Size:
  - text: 126668
  - data: 9800
  - bss: 107096

## Current Status

This version is ready for STM32F407 board-level network-only validation.

Expected UART output:

AI result: diff_cnt=0, max_abs_diff=0, cycles=xxxxx

## Notes

Do not add large RAM buffers before board validation.
Do not move activation pool to CCMRAM.
Sample arrays must remain static const int8_t.
