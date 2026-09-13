#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* Fuse W += scale * (alpha/rank) * B @ A into a loaded BF16 [rows, cols]
 * matrix. No-op when H3_LORA is unset. Reads H3_LORA as a comma-separated
 * list of PATH or PATH:SCALE. H3_LORA_SCALE is the default scale (1.0). */
int h3_lora_fuse(h3_gpu *gpu, h3_gpu_tensor *weight, const char *name,
                 uint64_t rows, uint64_t columns, char *error,
                 size_t error_size);

void h3_lora_reset(void);

#endif
