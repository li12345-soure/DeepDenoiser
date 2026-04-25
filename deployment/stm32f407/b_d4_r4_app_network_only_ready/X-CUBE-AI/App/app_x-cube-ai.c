
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "b_d4_r4.h"
#include "b_d4_r4_data.h"

/* USER CODE BEGIN includes */
#include "sample0_input_int8.h"
#include "sample0_output_int8.h"
/* USER CODE END includes */

#if SAMPLE0_INPUT_INT8_SIZE != AI_B_D4_R4_IN_1_SIZE_BYTES
#error "sample0 input size does not match network input size"
#endif

#if SAMPLE0_OUTPUT_INT8_SIZE != AI_B_D4_R4_OUT_1_SIZE_BYTES
#error "sample0 output size does not match network output size"
#endif

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_B_D4_R4_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_B_D4_R4_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_B_D4_R4_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_B_D4_R4_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_B_D4_R4_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_B_D4_R4_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_B_D4_R4_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_B_D4_R4_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_B_D4_R4_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle b_d4_r4 = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static uint32_t ai_run_cycles;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_b_d4_r4_create_and_init(&b_d4_r4, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_b_d4_r4_create_and_init");
    return -1;
  }

  ai_input = ai_b_d4_r4_inputs_get(b_d4_r4, NULL);
  ai_output = ai_b_d4_r4_outputs_get(b_d4_r4, NULL);

#if defined(AI_B_D4_R4_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_B_D4_R4_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_B_D4_R4_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_B_D4_R4_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_B_D4_R4_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_B_D4_R4_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;
  uint32_t start_cycles;

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  start_cycles = DWT->CYCCNT;

  batch = ai_b_d4_r4_run(b_d4_r4, ai_input, ai_output);
  ai_run_cycles = DWT->CYCCNT - start_cycles;
  if (batch != 1) {
    ai_log_err(ai_b_d4_r4_get_error(b_d4_r4),
        "ai_b_d4_r4_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
int acquire_and_process_data(ai_i8* data[])
{
  if ((data == NULL) || (data[0] == NULL)) {
    return -1;
  }

  memcpy(data[0], sample0_input_int8, SAMPLE0_INPUT_INT8_SIZE);

  return 0;
}

int post_process(ai_i8* data[])
{
  int diff_cnt = 0;
  int max_abs_diff = 0;

  if ((data == NULL) || (data[0] == NULL)) {
    return -1;
  }

  for (size_t i = 0; i < SAMPLE0_OUTPUT_INT8_SIZE; i++) {
    int diff = (int)data[0][i] - (int)sample0_output_int8[i];

    if (diff < 0) {
      diff = -diff;
    }

    if (diff != 0) {
      diff_cnt++;
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
      }
    }
  }

  printf("AI result: diff_cnt=%d, max_abs_diff=%d, cycles=%lu\r\n",
      diff_cnt, max_abs_diff, (unsigned long)ai_run_cycles);

  return 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  static uint8_t already_run = 0U;
  int res = -1;

  if (already_run != 0U) {
    return;
  }

  already_run = 1U;

  if (b_d4_r4) {

    /* 1 - acquire and pre-process input data */
    res = acquire_and_process_data(data_ins);
    /* 2 - process the data - call inference engine */
    if (res == 0)
      res = ai_run();
    /* 3- post-process the predictions */
    if (res == 0)
      res = post_process(data_outs);
  }

  if (res) {
    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
    ai_log_err(err, "Process has FAILED");
  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
