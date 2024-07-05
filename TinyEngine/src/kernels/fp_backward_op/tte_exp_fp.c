/* ----------------------------------------------------------------------
 * Project: Tiny Training Engine, MCUNetV3
 * Title:   tte_exp_fp.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *  - Chuang Gan, ganchuang@csail.mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp tte_exp(const uint16_t size, const float* input_data, float* output_data) {
	int i;

	for (i = 0; i < size; ++i) {
		output_data[i] = exp(input_data[i]);
	}

	/* Return to application */
	return STATE_SUCCESS_fp;
}
