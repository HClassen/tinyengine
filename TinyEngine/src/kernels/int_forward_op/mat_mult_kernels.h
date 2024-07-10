/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   mat_mult_kernels.h
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#ifndef TINYENGINE_MAT_MULT_KERNELS_H_
#define TINYENGINE_MAT_MULT_KERNELS_H_

#include <stdint.h>

#include "tinyengine/types.h"

q7_t *mat_mult_kernel_s8_s16(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
							 const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
							 const int16_t activation_min, const int16_t activation_max, const uint16_t num_col_a,
							 const int32_t *const output_bias, q7_t *out_0);

q7_t *mat_mult_unloop18_s8_s16(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
							   const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
							   const int16_t activation_min, const int16_t activation_max, const uint16_t num_col_a,
							   const int32_t *const output_bias, q7_t *out_0, q15_t *kbuf);

q7_t *mat_mult_s16_unloop8(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
						   const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
						   const int16_t activation_min, const int16_t activation_max, const uint16_t num_col_a,
						   const int32_t *const output_bias, q7_t *out_0, q15_t *kbuf);

q7_t *mat_mult_s16(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch, const int32_t *out_shift,
				   const int32_t *out_mult, const int32_t out_offset, const int16_t activation_min,
				   const int16_t activation_max, const uint16_t num_col_a, const int32_t *const output_bias,
				   q7_t *out_0, q15_t *kbuf);

q7_t *mat_mult_s16_funroll27(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
							 const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
							 const int16_t activation_min, const int16_t activation_max, const uint16_t num_col_a,
							 const int32_t *const output_bias, q7_t *out_0, q15_t *kbuf);

q7_t *mat_mult_s16_funroll8(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
							const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
							const int16_t activation_min, const int16_t activation_max, const uint16_t num_col_a,
							const int32_t *const output_bias, q7_t *out_0, q15_t *kbuf);

q7_t *mat_mult_s16_funroll16(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
							 const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
							 const int16_t activation_min, const int16_t activation_max, const uint16_t num_col_a,
							 const int32_t *const output_bias, q7_t *out_0, q15_t *kbuf);

q7_t *mat_mult_kernel_s8_s16_reordered_ch8(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
										   const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
										   const int16_t activation_min, const int16_t activation_max,
										   const uint16_t num_col_a, const int32_t *const output_bias, q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_ch16(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
											const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
											const int16_t activation_min, const int16_t activation_max,
											const uint16_t num_col_a, const int32_t *const output_bias, q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_ch24(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
											const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
											const int16_t activation_min, const int16_t activation_max,
											const uint16_t num_col_a, const int32_t *const output_bias, q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_ch48(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
											const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
											const int16_t activation_min, const int16_t activation_max,
											const uint16_t num_col_a, const int32_t *const output_bias, q7_t *out_0);

#endif /* TINYENGINE_MAT_MULT_KERNELS_H_ */
