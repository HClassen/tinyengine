/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   mat_mult_kernels_fpreq.h
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

#ifndef TINYENGINE_MAT_MULT_KERNELS_FPREQ_H_
#define TINYENGINE_MAT_MULT_KERNELS_FPREQ_H_

#include <stdint.h>

#include "tinyengine/types.h"

q7_t *mat_mult_kernel_s8_s16_reordered_ch48_fpreq(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
												  const float *scales, const int32_t out_offset,
												  const int16_t activation_min, const int16_t activation_max,
												  const uint16_t num_col_a, const int32_t *const output_bias,
												  q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_ch16_fpreq(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
												  const float *scales, const int32_t out_offset,
												  const int16_t activation_min, const int16_t activation_max,
												  const uint16_t num_col_a, const int32_t *const output_bias,
												  q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_ch8_fpreq(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
												 const float *scales, const int32_t out_offset,
												 const int16_t activation_min, const int16_t activation_max,
												 const uint16_t num_col_a, const int32_t *const output_bias,
												 q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_ch24_fpreq(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
												  const float *scales, const int32_t out_offset,
												  const int16_t activation_min, const int16_t activation_max,
												  const uint16_t num_col_a, const int32_t *const output_bias,
												  q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_fpreq(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
											 const float *scales, const int32_t out_offset,
											 const int16_t activation_min, const int16_t activation_max,
											 const uint16_t num_col_a, const int32_t *const output_bias, q7_t *out_0);

q7_t *mat_mult_kernel_s8_s16_reordered_fpreq_bitmask(const q7_t *input_a, const q15_t *input_b,
													 const uint16_t output_ch, const float *scales,
													 const int32_t out_offset, const int16_t activation_min,
													 const int16_t activation_max, const uint16_t num_col_a,
													 const int32_t *const output_bias, q7_t *out_0, q7_t *mask);

q7_t *mat_mult_kernel_s8_s16_reordered_fpreq_mask(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
												  const float *scales, const int32_t out_offset,
												  const int16_t activation_min, const int16_t activation_max,
												  const uint16_t num_col_a, const int32_t *const output_bias,
												  q7_t *out_0, q7_t *mask);

q7_t *mat_mult_kernel_s8_s16_reordered_fpreq_mask_partialCH(const q7_t *kernel_sram, const q7_t *kernel_flash,
															const uint16_t first_k_channel, const q15_t *input_b,
															const uint16_t output_ch, const float *scales,
															const int32_t out_offset, const int16_t activation_min,
															const int16_t activation_max, const uint16_t num_col_a,
															const int32_t *const output_bias, q7_t *out_0, q7_t *mask);

q7_t *mat_mult_kernel_s8_s16_reordered_fpreq_bitmask_partialCH(const q7_t *kernel_sram, const q7_t *kernel_flash,
                                                               const uint16_t first_k_channel, const q15_t *input_b,
	                                                           const uint16_t output_ch, const float *scales,
                                                               const int32_t out_offset, const int16_t activation_min,
	                                                           const int16_t activation_max, const uint16_t num_col_a,
                                                               const int32_t *const output_bias, q7_t *out_0, q7_t *mask);

q7_t *mat_mult_kernel_s8_s16_reordered_fpreq_mask_partialCH_Multiple2(const q7_t *kernel_sram, const q7_t *kernel_flash,
                                                                      const uint16_t first_k_channel, const q15_t *input_b,
	                                                                  const uint16_t output_ch, const float *scales,
                                                                      const int32_t out_offset, const int16_t activation_min,
	                                                                  const int16_t activation_max, const uint16_t num_col_a,
                                                                      const int32_t *const output_bias, q7_t *out_0, q7_t *mask);

q7_t *mat_mult_kernel_s8_s16_reordered_fpreq_bitmask_partialCH_Multiple2(const q7_t *kernel_sram, const q7_t *kernel_flash,
                                                                         const uint16_t first_k_channel, const q15_t *input_b,
	                                                                     const uint16_t output_ch, const float *scales,
                                                                         const int32_t out_offset, const int16_t activation_min,
	                                                                     const int16_t activation_max, const uint16_t num_col_a,
                                                                         const int32_t *const output_bias, q7_t *out_0, q7_t *mask);

q7_t *mat_mult_kernel3_input3_s8_s16_fpreq(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
										   const float *scales, const int32_t out_offset, const int16_t activation_min,
										   const int16_t activation_max, const uint16_t num_col_a,
										   const int32_t *const output_bias, q7_t *out_0, q15_t *kbuf);

#endif /* TINYENGINE_MAT_MULT_KERNELS_FPREQ_H_ */
