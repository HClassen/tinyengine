/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   stable_softmax.c
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

#include <float.h>
#include <math.h>

#include "tinyengine/types.h"
#include "tinyengine/base_ops.h"

tinyengine_status stable_softmax_inplace(float *input, const uint16_t length) {
	float max = FLT_MIN;
	float exp_sum = 0;

	for (uint16_t i = 0; i < length; i++) {
		if (input[i] > max)
            max = input[i];
	}

	// inplace update
	for (uint16_t i = 0; i < length; i++) {
		input[i] = exp(input[i] - max);
		exp_sum += input[i];
	}

	for (uint16_t i = 0; i < length; i++) {
		input[i] = input[i] / exp_sum;
	}

    return STATE_SUCCESS;
}
