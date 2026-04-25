/**
  ******************************************************************************
  * @file    b_d4_r4.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-04-25T11:24:29+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "b_d4_r4.h"
#include "b_d4_r4_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_b_d4_r4
 
#undef AI_B_D4_R4_MODEL_SIGNATURE
#define AI_B_D4_R4_MODEL_SIGNATURE     "0x5cb97cd66120483049988cdb1d0751c9"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-04-25T11:24:29+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_B_D4_R4_N_BATCHES
#define AI_B_D4_R4_N_BATCHES         (1)

static ai_ptr g_b_d4_r4_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_b_d4_r4_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  X_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 12462, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24924, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24924, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6464, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12928, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3264, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6528, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1664, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3328, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_upsample_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11424, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 17280, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6656, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  slice_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6528, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  concat_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13056, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16960, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6528, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_upsample_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24240, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 29952, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13056, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  slice_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12928, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  concat_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25856, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 29664, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12928, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_upsample_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 49848, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25856, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  slice_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24924, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  concat_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 49848, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24924, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 12462, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 2, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 272, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 488, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 488, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 832, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1552, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2816, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5408, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6496, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6496, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2992, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2992, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 920, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 920, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(X_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2968551814556122f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020190684124827385f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041976992040872574f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0957183837890625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12567079067230225f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0027081910520792007f, 0.0034142464864999056f, 0.0032399685587733984f, 0.003219534642994404f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01854962855577469f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020190684124827385f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001281603705137968f, 0.0012465895852074027f, 0.0014284306671470404f, 0.0013264879817143083f, 0.0012667571427300572f, 0.0010897605679929256f, 0.001215716591104865f, 0.0011122683063149452f, 0.0013484688242897391f, 0.0012904283357784152f, 0.0011940601980313659f, 0.0013687827158719301f, 0.001249767025001347f, 0.0013316705590113997f, 0.0012761359103024006f, 0.0012264061952009797f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_12_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041976992040872574f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_12_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01854962855577469f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_12_upsample_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01854962855577469f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_12_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002553665079176426f, 0.0023528332822024822f, 0.002402716549113393f, 0.002402627607807517f, 0.002500304253771901f, 0.0023448788560926914f, 0.0025096663739532232f, 0.002922190586104989f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039307571947574615f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041976992040872574f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0014748512767255306f, 0.0017275831196457148f, 0.0020214824471622705f, 0.0017029073787853122f, 0.001784265274181962f, 0.0015783687122166157f, 0.002205694792792201f, 0.0018529605586081743f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0957183837890625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_16_upsample_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039307571947574615f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_16_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0037729674950242043f, 0.0035523874685168266f, 0.0036739888601005077f, 0.0039538186974823475f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_19_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05068913847208023f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_19_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002586173824965954f, 0.0030803789850324392f, 0.002221949165686965f, 0.003066725330427289f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0957183837890625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0028436218854039907f, 0.00400205422192812f, 0.0028630848973989487f, 0.0025299428962171078f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10083365440368652f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_20_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 2,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00734016252681613f, 0.004874160513281822f),
    AI_PACK_INTQ_ZP(0, 0)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.044187434017658234f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003102188464254141f, 0.0037743523716926575f, 0.003931507468223572f, 0.0037767537869513035f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_3_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041976992040872574f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_3_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0027676301542669535f, 0.002390615176409483f, 0.003323687007650733f, 0.0030507538467645645f, 0.0028304459992796183f, 0.003050238359719515f, 0.0029012211598455906f, 0.0036590909585356712f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.032374169677495956f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002509117592126131f, 0.002582860179245472f, 0.0027622710913419724f, 0.0027871110942214727f, 0.0026085292920470238f, 0.0025280879344791174f, 0.002593725686892867f, 0.002345466520637274f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020190684124827385f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0023790374398231506f, 0.0021188626997172832f, 0.0022896025329828262f, 0.0024165059439837933f, 0.0024863991420716047f, 0.0021904869936406612f, 0.0021528860088437796f, 0.0022165372502058744f, 0.0024396239314228296f, 0.0024005717132240534f, 0.0020156113896518946f, 0.0023790050763636827f, 0.0025349422357976437f, 0.0021863810252398252f, 0.0023895243648439646f, 0.0023093547206372023f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01796143501996994f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_6_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019299488049000502f, 0.0016298305708914995f, 0.0016721909632906318f, 0.0015916161937639117f, 0.0018748905276879668f, 0.0012980158207938075f, 0.0015641874633729458f, 0.0017808780539780855f, 0.0018572076223790646f, 0.0018165655201300979f, 0.0016587453428655863f, 0.0015724038239568472f, 0.0017902750987559557f, 0.0015402983408421278f, 0.001999090425670147f, 0.001450242823921144f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013216419145464897f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01796143501996994f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016867014346644282f, 0.0016436964506283402f, 0.0018167223315685987f, 0.0016006670193746686f, 0.0018267066916450858f, 0.0015414571389555931f, 0.0017375530442222953f, 0.0018905044998973608f, 0.0016624246491119266f, 0.0015734172193333507f, 0.0018023209413513541f, 0.0015971997054293752f, 0.0016939722700044513f, 0.0017748140962794423f, 0.001801581820473075f, 0.0018291017040610313f, 0.0015544843627139926f, 0.0017866905545815825f, 0.0013483127113431692f, 0.001862811972387135f, 0.001697107800282538f, 0.0016422714106738567f, 0.0015668687410652637f, 0.001774936681613326f, 0.0016619719099253416f, 0.0017236750572919846f, 0.0017152744112536311f, 0.0011989227496087551f, 0.0018061483278870583f, 0.0017410562140867114f, 0.0017709676176309586f, 0.0018444294109940529f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020190684124827385f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013216419145464897f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_upsample_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013216419145464897f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019392293179407716f, 0.0019348212517797947f, 0.0018324641278013587f, 0.0018320708768442273f, 0.0019220843678340316f, 0.0020280920434743166f, 0.0018042318988591433f, 0.0018008177867159247f, 0.0018962097819894552f, 0.001868344028480351f, 0.001952304970473051f, 0.0019726695027202368f, 0.0017716547008603811f, 0.0016787999775260687f, 0.0018013534136116505f, 0.0017681760946288705f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041976992040872574f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_17_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0957183837890625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_9_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020190684124827385f),
    AI_PACK_INTQ_ZP(-128)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  X_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 201, 31), AI_STRIDE_INIT(4, 1, 1, 2, 402),
  1, &X_output_array, &X_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  concat_10_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 51, 8), AI_STRIDE_INIT(4, 1, 1, 32, 1632),
  1, &concat_10_output_array, &concat_10_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  concat_14_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 101, 16), AI_STRIDE_INIT(4, 1, 1, 16, 1616),
  1, &concat_14_output_array, &concat_14_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  concat_18_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 201, 31), AI_STRIDE_INIT(4, 1, 1, 8, 1608),
  1, &concat_18_output_array, &concat_18_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_0_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 201, 31), AI_STRIDE_INIT(4, 1, 1, 4, 804),
  1, &conv2d_0_output_array, &conv2d_0_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 272, 1, 1), AI_STRIDE_INIT(4, 1, 1, 272, 272),
  1, &conv2d_0_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 2, 3, 3, 4), AI_STRIDE_INIT(4, 1, 2, 8, 24),
  1, &conv2d_0_weights_array, &conv2d_0_weights_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_11_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 51, 8), AI_STRIDE_INIT(4, 1, 1, 16, 816),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_pad_before_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 53, 10), AI_STRIDE_INIT(4, 1, 1, 32, 1696),
  1, &conv2d_11_pad_before_output_array, &conv2d_11_pad_before_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_scratch0, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 6496, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6496, 6496),
  1, &conv2d_11_scratch0_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 1, 32, 512, 1536),
  1, &conv2d_11_weights_array, &conv2d_11_weights_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_12_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 102, 16), AI_STRIDE_INIT(4, 1, 1, 8, 816),
  1, &conv2d_12_output_array, &conv2d_12_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_pad_before_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 104, 18), AI_STRIDE_INIT(4, 1, 1, 16, 1664),
  1, &conv2d_12_pad_before_output_array, &conv2d_12_pad_before_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 2992, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2992, 2992),
  1, &conv2d_12_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_upsample_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 101, 15), AI_STRIDE_INIT(4, 1, 1, 16, 1616),
  1, &conv2d_12_upsample_output_array, &conv2d_12_upsample_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_weights, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 8), AI_STRIDE_INIT(4, 1, 16, 128, 384),
  1, &conv2d_12_weights_array, &conv2d_12_weights_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_15_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 101, 16), AI_STRIDE_INIT(4, 1, 1, 8, 808),
  1, &conv2d_15_output_array, &conv2d_15_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_pad_before_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 103, 18), AI_STRIDE_INIT(4, 1, 1, 16, 1648),
  1, &conv2d_15_pad_before_output_array, &conv2d_15_pad_before_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_scratch0, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 2992, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2992, 2992),
  1, &conv2d_15_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_weights, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 8), AI_STRIDE_INIT(4, 1, 16, 128, 384),
  1, &conv2d_15_weights_array, &conv2d_15_weights_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_16_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 202, 32), AI_STRIDE_INIT(4, 1, 1, 4, 808),
  1, &conv2d_16_output_array, &conv2d_16_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_scratch0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 920, 1, 1), AI_STRIDE_INIT(4, 1, 1, 920, 920),
  1, &conv2d_16_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_upsample_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 201, 31), AI_STRIDE_INIT(4, 1, 1, 8, 1608),
  1, &conv2d_16_upsample_output_array, &conv2d_16_upsample_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_weights, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 4), AI_STRIDE_INIT(4, 1, 8, 32, 96),
  1, &conv2d_16_weights_array, &conv2d_16_weights_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_19_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 201, 31), AI_STRIDE_INIT(4, 1, 1, 4, 804),
  1, &conv2d_19_output_array, &conv2d_19_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_scratch0, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 920, 1, 1), AI_STRIDE_INIT(4, 1, 1, 920, 920),
  1, &conv2d_19_scratch0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_weights, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 4), AI_STRIDE_INIT(4, 1, 8, 32, 96),
  1, &conv2d_19_weights_array, &conv2d_19_weights_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 201, 31), AI_STRIDE_INIT(4, 1, 1, 4, 804),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 488, 1, 1), AI_STRIDE_INIT(4, 1, 1, 488, 488),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 4, 3, 3, 4), AI_STRIDE_INIT(4, 1, 4, 16, 48),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv2d_20_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 201, 31), AI_STRIDE_INIT(4, 1, 1, 2, 402),
  1, &conv2d_20_output_array, &conv2d_20_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_scratch0, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 1, 1, 36, 36),
  1, &conv2d_20_scratch0_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_weights, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 4, 1, 1, 2), AI_STRIDE_INIT(4, 1, 4, 8, 8),
  1, &conv2d_20_weights_array, &conv2d_20_weights_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 101, 16), AI_STRIDE_INIT(4, 1, 1, 4, 404),
  1, &conv2d_2_output_array, &conv2d_2_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 488, 1, 1), AI_STRIDE_INIT(4, 1, 1, 488, 488),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 4, 3, 3, 4), AI_STRIDE_INIT(4, 1, 4, 16, 48),
  1, &conv2d_2_weights_array, &conv2d_2_weights_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_3_bias_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 101, 16), AI_STRIDE_INIT(4, 1, 1, 8, 808),
  1, &conv2d_3_output_array, &conv2d_3_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_scratch0, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 832, 1, 1), AI_STRIDE_INIT(4, 1, 1, 832, 832),
  1, &conv2d_3_scratch0_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 4, 3, 3, 8), AI_STRIDE_INIT(4, 1, 4, 32, 96),
  1, &conv2d_3_weights_array, &conv2d_3_weights_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 51, 8), AI_STRIDE_INIT(4, 1, 1, 8, 408),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 1552, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1552, 1552),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 8), AI_STRIDE_INIT(4, 1, 8, 64, 192),
  1, &conv2d_4_weights_array, &conv2d_4_weights_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_bias, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_5_bias_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 51, 8), AI_STRIDE_INIT(4, 1, 1, 16, 816),
  1, &conv2d_5_output_array, &conv2d_5_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch0, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 2816, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2816, 2816),
  1, &conv2d_5_scratch0_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_weights, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 16), AI_STRIDE_INIT(4, 1, 8, 128, 384),
  1, &conv2d_5_weights_array, &conv2d_5_weights_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_bias, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_6_bias_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 26, 4), AI_STRIDE_INIT(4, 1, 1, 16, 416),
  1, &conv2d_6_output_array, &conv2d_6_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_scratch0, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 5408, 1, 1), AI_STRIDE_INIT(4, 1, 1, 5408, 5408),
  1, &conv2d_6_scratch0_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_weights, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 16), AI_STRIDE_INIT(4, 1, 16, 256, 768),
  1, &conv2d_6_weights_array, &conv2d_6_weights_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 26, 4), AI_STRIDE_INIT(4, 1, 1, 32, 832),
  1, &conv2d_7_output_array, &conv2d_7_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_pad_before_output, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 28, 6), AI_STRIDE_INIT(4, 1, 1, 16, 448),
  1, &conv2d_7_pad_before_output_array, &conv2d_7_pad_before_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 6144, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6144, 6144),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 1, 16, 512, 1536),
  1, &conv2d_7_weights_array, &conv2d_7_weights_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_8_bias_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 52, 8), AI_STRIDE_INIT(4, 1, 1, 16, 832),
  1, &conv2d_8_output_array, &conv2d_8_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_pad_before_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 54, 10), AI_STRIDE_INIT(4, 1, 1, 32, 1728),
  1, &conv2d_8_pad_before_output_array, &conv2d_8_pad_before_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch0, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 6496, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6496, 6496),
  1, &conv2d_8_scratch0_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_upsample_output, AI_STATIC,
  70, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 51, 7), AI_STRIDE_INIT(4, 1, 1, 32, 1632),
  1, &conv2d_8_upsample_output_array, &conv2d_8_upsample_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 1, 32, 512, 1536),
  1, &conv2d_8_weights_array, &conv2d_8_weights_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  slice_13_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 101, 16), AI_STRIDE_INIT(4, 1, 1, 8, 808),
  1, &slice_13_output_array, &slice_13_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  slice_17_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 201, 31), AI_STRIDE_INIT(4, 1, 1, 4, 804),
  1, &slice_17_output_array, &slice_17_output_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  slice_9_output, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 51, 8), AI_STRIDE_INIT(4, 1, 1, 16, 816),
  1, &slice_9_output_array, &slice_9_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_20_weights, &conv2d_20_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_20_layer, 20,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &conv2d_20_chain,
  NULL, &conv2d_20_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_19_weights, &conv2d_19_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_19_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_19_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_19_chain,
  NULL, &conv2d_20_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_1_output, &slice_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_18_layer, 18,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_18_chain,
  NULL, &conv2d_19_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_u8 slice_17_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_17_axes, AI_ARRAY_FORMAT_U8,
    slice_17_axes_data, slice_17_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_17_starts_data[] = { 0, 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_17_starts, AI_ARRAY_FORMAT_S16,
    slice_17_starts_data, slice_17_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_17_ends_data[] = { 31, 201, 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_17_ends, AI_ARRAY_FORMAT_S16,
    slice_17_ends_data, slice_17_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_17_layer, 17,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_17_chain,
  NULL, &concat_18_layer, AI_STATIC, 
  .axes = &slice_17_axes, 
  .starts = &slice_17_starts, 
  .ends = &slice_17_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_16_weights, &conv2d_16_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_16_layer, 16,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_16_chain,
  NULL, &slice_17_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 2, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float conv2d_16_upsample_scales_data[] = { 2, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_16_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    conv2d_16_upsample_scales_data, conv2d_16_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_16_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_16_upsample_layer, 16,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros_is8os8,
  &conv2d_16_upsample_chain,
  NULL, &conv2d_16_layer, AI_STATIC, 
  .scales = &conv2d_16_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_15_weights, &conv2d_15_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_15_layer, 15,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_3x3_sssa8_ch,
  &conv2d_15_chain,
  NULL, &conv2d_16_upsample_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_15_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_15_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_15_pad_before_value_data, conv2d_15_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_15_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_15_pad_before_layer, 15,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_15_pad_before_chain,
  NULL, &conv2d_15_layer, AI_STATIC, 
  .value = &conv2d_15_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_3_output, &slice_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_14_layer, 14,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_14_chain,
  NULL, &conv2d_15_pad_before_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_u8 slice_13_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_13_axes, AI_ARRAY_FORMAT_U8,
    slice_13_axes_data, slice_13_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_13_starts_data[] = { 0, 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_13_starts, AI_ARRAY_FORMAT_S16,
    slice_13_starts_data, slice_13_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_13_ends_data[] = { 16, 101, 8 };
AI_ARRAY_OBJ_DECLARE(
    slice_13_ends, AI_ARRAY_FORMAT_S16,
    slice_13_ends_data, slice_13_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_13_layer, 13,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_13_chain,
  NULL, &concat_14_layer, AI_STATIC, 
  .axes = &slice_13_axes, 
  .starts = &slice_13_starts, 
  .ends = &slice_13_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_12_weights, &conv2d_12_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_12_layer, 12,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_3x3_sssa8_ch,
  &conv2d_12_chain,
  NULL, &slice_13_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_12_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_12_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_12_pad_before_value_data, conv2d_12_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_12_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_12_pad_before_layer, 12,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_12_pad_before_chain,
  NULL, &conv2d_12_layer, AI_STATIC, 
  .value = &conv2d_12_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 2, 2, 1, 1), 
)


AI_STATIC_CONST ai_float conv2d_12_upsample_scales_data[] = { 2, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_12_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    conv2d_12_upsample_scales_data, conv2d_12_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_12_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_12_upsample_layer, 12,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros_is8os8,
  &conv2d_12_upsample_chain,
  NULL, &conv2d_12_pad_before_layer, AI_STATIC, 
  .scales = &conv2d_12_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_11_weights, &conv2d_11_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_3x3_sssa8_ch,
  &conv2d_11_chain,
  NULL, &conv2d_12_upsample_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_11_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_11_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_11_pad_before_value_data, conv2d_11_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_pad_before_layer, 11,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_11_pad_before_chain,
  NULL, &conv2d_11_layer, AI_STATIC, 
  .value = &conv2d_11_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_5_output, &slice_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_10_layer, 10,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_10_chain,
  NULL, &conv2d_11_pad_before_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_u8 slice_9_axes_data[] = { 0, 1, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_9_axes, AI_ARRAY_FORMAT_U8,
    slice_9_axes_data, slice_9_axes_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_9_starts_data[] = { 0, 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_9_starts, AI_ARRAY_FORMAT_S16,
    slice_9_starts_data, slice_9_starts_data, 3, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_9_ends_data[] = { 8, 51, 16 };
AI_ARRAY_OBJ_DECLARE(
    slice_9_ends, AI_ARRAY_FORMAT_S16,
    slice_9_ends_data, slice_9_ends_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_9_layer, 9,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_9_chain,
  NULL, &concat_10_layer, AI_STATIC, 
  .axes = &slice_9_axes, 
  .starts = &slice_9_starts, 
  .ends = &slice_9_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_3x3_sssa8_ch,
  &conv2d_8_chain,
  NULL, &slice_9_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_8_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_8_pad_before_value_data, conv2d_8_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_pad_before_layer, 8,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_8_pad_before_chain,
  NULL, &conv2d_8_layer, AI_STATIC, 
  .value = &conv2d_8_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 2, 2, 1, 1), 
)


AI_STATIC_CONST ai_float conv2d_8_upsample_scales_data[] = { 2, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    conv2d_8_upsample_scales_data, conv2d_8_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_upsample_layer, 8,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros_is8os8,
  &conv2d_8_upsample_chain,
  NULL, &conv2d_8_pad_before_layer, AI_STATIC, 
  .scales = &conv2d_8_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_3x3_sssa8_ch,
  &conv2d_7_chain,
  NULL, &conv2d_8_upsample_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_7_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_7_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_7_pad_before_value_data, conv2d_7_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_pad_before_layer, 7,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_7_pad_before_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .value = &conv2d_7_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_6_weights, &conv2d_6_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_6_layer, 6,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_6_chain,
  NULL, &conv2d_7_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 2, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_5_weights, &conv2d_5_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_5_layer, 5,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_5_chain,
  NULL, &conv2d_6_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_4_chain,
  NULL, &conv2d_5_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 2, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 3,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_3_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_2_chain,
  NULL, &conv2d_3_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_1_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &X_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 0,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_0_chain,
  NULL, &conv2d_1_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 21992, 1, 1),
    21992, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 102132, 1, 1),
    102132, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_B_D4_R4_IN_NUM, &X_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_B_D4_R4_OUT_NUM, &conv2d_20_output),
  &conv2d_0_layer, 0xceee895b, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 21992, 1, 1),
      21992, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 102132, 1, 1),
      102132, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_B_D4_R4_IN_NUM, &X_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_B_D4_R4_OUT_NUM, &conv2d_20_output),
  &conv2d_0_layer, 0xceee895b, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool b_d4_r4_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_b_d4_r4_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    X_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 67980);
    X_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 67980);
    conv2d_0_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 80444);
    conv2d_0_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 80444);
    conv2d_0_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 53900);
    conv2d_0_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 53900);
    conv2d_1_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 78824);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 78824);
    conv2d_1_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 52284);
    conv2d_1_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 52284);
    conv2d_2_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 80228);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 80228);
    conv2d_2_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 45820);
    conv2d_2_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 45820);
    conv2d_3_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_3_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_3_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 37724);
    conv2d_3_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 37724);
    conv2d_4_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 50732);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 50732);
    conv2d_4_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77452);
    conv2d_4_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77452);
    conv2d_5_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 80716);
    conv2d_5_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 80716);
    conv2d_5_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 31196);
    conv2d_5_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 31196);
    conv2d_6_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 78124);
    conv2d_6_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 78124);
    conv2d_6_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 29532);
    conv2d_6_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 29532);
    conv2d_7_pad_before_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 28508);
    conv2d_7_pad_before_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 28508);
    conv2d_7_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_7_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_7_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 83352);
    conv2d_7_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 83352);
    conv2d_8_upsample_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 19772);
    conv2d_8_upsample_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 19772);
    conv2d_8_pad_before_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_8_pad_before_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_8_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 24700);
    conv2d_8_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 24700);
    conv2d_8_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 18044);
    conv2d_8_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 18044);
    slice_9_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    slice_9_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    concat_10_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 18140);
    concat_10_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 18140);
    conv2d_11_pad_before_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 14236);
    conv2d_11_pad_before_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 14236);
    conv2d_11_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 31196);
    conv2d_11_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 31196);
    conv2d_11_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_11_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_12_upsample_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 13484);
    conv2d_12_upsample_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 13484);
    conv2d_12_pad_before_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 7772);
    conv2d_12_pad_before_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 7772);
    conv2d_12_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 91496);
    conv2d_12_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 91496);
    conv2d_12_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 6956);
    conv2d_12_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 6956);
    slice_13_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    slice_13_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    concat_14_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 11868);
    concat_14_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 11868);
    conv2d_15_pad_before_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 8060);
    conv2d_15_pad_before_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 8060);
    conv2d_15_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 5068);
    conv2d_15_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 5068);
    conv2d_15_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_15_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_16_upsample_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 2436);
    conv2d_16_upsample_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 2436);
    conv2d_16_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_16_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    conv2d_16_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 0);
    conv2d_16_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 0);
    slice_17_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    slice_17_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 77208);
    concat_18_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 0);
    concat_18_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 0);
    conv2d_19_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 49848);
    conv2d_19_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 49848);
    conv2d_19_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 50768);
    conv2d_19_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 50768);
    conv2d_20_scratch0_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 0);
    conv2d_20_scratch0_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 0);
    conv2d_20_output_array.data = AI_PTR(g_b_d4_r4_activations_map[0] + 36);
    conv2d_20_output_array.data_start = AI_PTR(g_b_d4_r4_activations_map[0] + 36);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool b_d4_r4_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_b_d4_r4_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 0);
    conv2d_0_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 0);
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 72);
    conv2d_0_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 72);
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 88);
    conv2d_1_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 88);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 232);
    conv2d_1_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 232);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 248);
    conv2d_2_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 248);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 392);
    conv2d_2_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 392);
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 408);
    conv2d_3_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 408);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 696);
    conv2d_3_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 696);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 728);
    conv2d_4_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 728);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 1304);
    conv2d_4_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 1304);
    conv2d_5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 1336);
    conv2d_5_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 1336);
    conv2d_5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 2488);
    conv2d_5_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 2488);
    conv2d_6_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 2552);
    conv2d_6_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 2552);
    conv2d_6_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 4856);
    conv2d_6_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 4856);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 4920);
    conv2d_7_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 4920);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 9528);
    conv2d_7_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 9528);
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 9656);
    conv2d_8_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 9656);
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 14264);
    conv2d_8_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 14264);
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 14328);
    conv2d_11_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 14328);
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 18936);
    conv2d_11_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 18936);
    conv2d_12_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 19000);
    conv2d_12_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 19000);
    conv2d_12_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 20152);
    conv2d_12_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 20152);
    conv2d_15_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 20184);
    conv2d_15_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 20184);
    conv2d_15_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 21336);
    conv2d_15_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 21336);
    conv2d_16_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 21368);
    conv2d_16_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 21368);
    conv2d_16_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 21656);
    conv2d_16_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 21656);
    conv2d_19_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 21672);
    conv2d_19_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 21672);
    conv2d_19_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 21960);
    conv2d_19_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 21960);
    conv2d_20_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_weights_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 21976);
    conv2d_20_weights_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 21976);
    conv2d_20_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_bias_array.data = AI_PTR(g_b_d4_r4_weights_map[0] + 21984);
    conv2d_20_bias_array.data_start = AI_PTR(g_b_d4_r4_weights_map[0] + 21984);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_b_d4_r4_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_B_D4_R4_MODEL_NAME,
      .model_signature   = AI_B_D4_R4_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 14712726,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xceee895b,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_b_d4_r4_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_B_D4_R4_MODEL_NAME,
      .model_signature   = AI_B_D4_R4_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 14712726,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xceee895b,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_b_d4_r4_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_b_d4_r4_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_b_d4_r4_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_b_d4_r4_create(network, AI_B_D4_R4_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_b_d4_r4_data_params_get(&params) != true) {
    err = ai_b_d4_r4_get_error(*network);
    return err;
  }
#if defined(AI_B_D4_R4_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_B_D4_R4_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_b_d4_r4_init(*network, &params) != true) {
    err = ai_b_d4_r4_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_b_d4_r4_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_b_d4_r4_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_b_d4_r4_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_b_d4_r4_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= b_d4_r4_configure_weights(net_ctx, params);
  ok &= b_d4_r4_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_b_d4_r4_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_b_d4_r4_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_B_D4_R4_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

