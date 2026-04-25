/**
  ******************************************************************************
  * @file    b_d4_r4_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-04-25T11:24:29+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef B_D4_R4_DATA_PARAMS_H
#define B_D4_R4_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_B_D4_R4_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_b_d4_r4_data_weights_params[1]))
*/

#define AI_B_D4_R4_DATA_CONFIG               (NULL)


#define AI_B_D4_R4_DATA_ACTIVATIONS_SIZES \
  { 102132, }
#define AI_B_D4_R4_DATA_ACTIVATIONS_SIZE     (102132)
#define AI_B_D4_R4_DATA_ACTIVATIONS_COUNT    (1)
#define AI_B_D4_R4_DATA_ACTIVATION_1_SIZE    (102132)



#define AI_B_D4_R4_DATA_WEIGHTS_SIZES \
  { 21992, }
#define AI_B_D4_R4_DATA_WEIGHTS_SIZE         (21992)
#define AI_B_D4_R4_DATA_WEIGHTS_COUNT        (1)
#define AI_B_D4_R4_DATA_WEIGHT_1_SIZE        (21992)



#define AI_B_D4_R4_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_b_d4_r4_activations_table[1])

extern ai_handle g_b_d4_r4_activations_table[1 + 2];



#define AI_B_D4_R4_DATA_WEIGHTS_TABLE_GET() \
  (&g_b_d4_r4_weights_table[1])

extern ai_handle g_b_d4_r4_weights_table[1 + 2];


#endif    /* B_D4_R4_DATA_PARAMS_H */
