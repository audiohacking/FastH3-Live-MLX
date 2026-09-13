#include "h3_lora.h"

#include "h3_safetensors.h"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_LORA_MAX 8
#define H3_LORA_NAME 192

typedef struct {
    char *name;
    float *data;
    int ndim;
    uint64_t shape[2];
    size_t elements;
} h3_lora_tensor;

typedef struct {
    char path[4096];
    float scale;
    h3_st_header header;
    h3_lora_tensor *tensors;
    size_t tensor_count;
    int loaded;
    int fused;
    int missing;
} h3_lora_pack;

static h3_lora_pack g_packs[H3_LORA_MAX];
static size_t g_pack_count;
static int g_ready;
static int g_failed;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = ((uint32_t)value) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void copy_name(char *destination, size_t size, const char *source) {
    if (!destination || !size) return;
    snprintf(destination, size, "%s", source ? source : "");
}

static const char *strip_prefix(const char *name) {
    static const char *prefixes[] = {
        "base_model.model.",
        "base_model.",
        "diffusion_model.",
        "model.diffusion_model.",
        "transformer.",
        "unet.",
        "lora_unet_",
        NULL,
    };
    for (size_t index = 0; prefixes[index]; index++) {
        size_t length = strlen(prefixes[index]);
        if (strncmp(name, prefixes[index], length) == 0) return name + length;
    }
    return name;
}

static void rewrite_prefix(char *buffer, const char *from, const char *to) {
    size_t from_n = strlen(from);
    if (strncmp(buffer, from, from_n) != 0) return;
    size_t to_n = strlen(to);
    size_t rest = strlen(buffer + from_n);
    if (to_n + rest + 1 > H3_LORA_NAME) return;
    memmove(buffer + to_n, buffer + from_n, rest + 1);
    memcpy(buffer, to, to_n);
}

static void canon_key(const char *source, char *destination, size_t size) {
    const char *text = strip_prefix(source);
    char buffer[H3_LORA_NAME];
    copy_name(buffer, sizeof(buffer), text);
    /* PEFT: foo.lora_A.default.weight → foo.lora_A.weight */
    char *def = strstr(buffer, ".default.");
    if (def) {
        memmove(def, def + 8, strlen(def + 8) + 1);
    }
    /* Diffusers PEFT uses transformer_blocks / refiner_blocks; h3.c uses blocks. */
    rewrite_prefix(buffer, "transformer_blocks.", "blocks.");
    rewrite_prefix(buffer, "token_refiner.refiner_blocks.",
                   "token_refiner.blocks.");
    copy_name(destination, size, buffer);
}

static int ends_with(const char *text, const char *suffix) {
    size_t n = strlen(text);
    size_t m = strlen(suffix);
    return n >= m && strcmp(text + n - m, suffix) == 0;
}

static int stem_before(const char *name, const char *suffix, char *stem,
                       size_t stem_size) {
    if (!ends_with(name, suffix)) return 0;
    size_t keep = strlen(name) - strlen(suffix);
    if (keep + 8 >= stem_size) return 0;
    memcpy(stem, name, keep);
    stem[keep] = '\0';
    return 1;
}

static h3_lora_tensor *find_tensor(h3_lora_pack *pack, const char *name) {
    for (size_t index = 0; index < pack->tensor_count; index++) {
        if (!strcmp(pack->tensors[index].name, name)) return &pack->tensors[index];
    }
    return NULL;
}

static int load_pack_tensors(h3_lora_pack *pack, char *error, size_t error_size) {
    if (!h3_st_read_header(pack->path, &pack->header, error, error_size)) return 0;
    pack->tensors = calloc(pack->header.tensor_count, sizeof(*pack->tensors));
    if (!pack->tensors) {
        fail(error, error_size, "out of memory loading LoRA %s", pack->path);
        return 0;
    }
    pack->tensor_count = pack->header.tensor_count;
    for (size_t index = 0; index < pack->header.tensor_count; index++) {
        const h3_st_tensor *src = &pack->header.tensors[index];
        h3_lora_tensor *dst = &pack->tensors[index];
        char canon[H3_LORA_NAME];
        canon_key(src->name, canon, sizeof(canon));
        dst->name = strdup(canon);
        dst->ndim = src->ndim > 2 ? 2 : src->ndim;
        dst->shape[0] = src->ndim >= 1 ? src->shape[0] : 1;
        dst->shape[1] = src->ndim >= 2 ? src->shape[1] : 1;
        dst->elements = (size_t)h3_st_tensor_elements(src);
        if (!dst->name) {
            fail(error, error_size, "out of memory copying LoRA tensor name");
            return 0;
        }
        size_t bytes = dst->elements * h3_dtype_size(src->dtype);
        void *raw = malloc(bytes ? bytes : 1);
        if (!raw) {
            fail(error, error_size, "out of memory reading LoRA tensor %s",
                 src->name);
            return 0;
        }
        if (!h3_st_read_data(&pack->header, src, raw, bytes, error, error_size)) {
            free(raw);
            return 0;
        }
        dst->data = malloc(dst->elements * sizeof(float));
        if (!dst->data) {
            free(raw);
            fail(error, error_size, "out of memory converting LoRA tensor %s",
                 src->name);
            return 0;
        }
        if (src->dtype == H3_DTYPE_F32) {
            memcpy(dst->data, raw, dst->elements * sizeof(float));
        } else if (src->dtype == H3_DTYPE_BF16) {
            const uint16_t *bf = raw;
            for (size_t i = 0; i < dst->elements; i++) {
                dst->data[i] = bf16_to_f32(bf[i]);
            }
        } else if (src->dtype == H3_DTYPE_F16) {
            /* IEEE F16 is uncommon in these adapters; reject rather than guess. */
            free(raw);
            fail(error, error_size,
                 "LoRA tensor %s is F16; use BF16 or F32 weights", src->name);
            return 0;
        } else {
            free(raw);
            fail(error, error_size, "unsupported LoRA dtype for %s (%s)",
                 src->name, h3_dtype_name(src->dtype));
            return 0;
        }
        free(raw);
    }
    pack->loaded = 1;
    return 1;
}

static int is_dit_matrix(const char *name) {
    const char *text = strip_prefix(name);
    return strncmp(text, "blocks.", 7) == 0 ||
           strncmp(text, "token_refiner.", 14) == 0 ||
           strncmp(text, "final_layer.", 12) == 0;
}

static void h3_lora_report(void) {
    for (size_t index = 0; index < g_pack_count; index++) {
        h3_lora_pack *pack = &g_packs[index];
        fprintf(stderr, "h3: LoRA summary %s  fused=%d  unmatched_dit=%d\n",
                pack->path, pack->fused, pack->missing);
    }
}

static int parse_packs(char *error, size_t error_size) {
    const char *raw = getenv("H3_LORA");
    if (!raw || !*raw) {
        g_ready = 1;
        return 1;
    }
    float default_scale = 1.0f;
    const char *scale_env = getenv("H3_LORA_SCALE");
    if (scale_env && *scale_env) {
        char *end = NULL;
        float parsed = strtof(scale_env, &end);
        if (end != scale_env && isfinite(parsed) && parsed >= 0.0f) {
            default_scale = parsed;
        }
    }
    char copy[8192];
    copy_name(copy, sizeof(copy), raw);
    char *cursor = copy;
    while (*cursor && g_pack_count < H3_LORA_MAX) {
        while (*cursor && (isspace((unsigned char)*cursor) || *cursor == ',')) {
            cursor++;
        }
        if (!*cursor) break;
        char *item = cursor;
        while (*cursor && *cursor != ',') cursor++;
        if (*cursor) *cursor++ = '\0';
        char *colon = strrchr(item, ':');
        float scale = default_scale;
        if (colon && colon != item && colon[1] &&
            (colon[1] == '.' || isdigit((unsigned char)colon[1]))) {
            char *end = NULL;
            float parsed = strtof(colon + 1, &end);
            if (end != colon + 1 && isfinite(parsed) && parsed >= 0.0f) {
                *colon = '\0';
                scale = parsed;
            }
        }
        if (!*item) continue;
        h3_lora_pack *pack = &g_packs[g_pack_count++];
        memset(pack, 0, sizeof(*pack));
        copy_name(pack->path, sizeof(pack->path), item);
        pack->scale = scale;
        if (!load_pack_tensors(pack, error, error_size)) {
            g_failed = 1;
            return 0;
        }
        fprintf(stderr, "h3: LoRA %s  tensors=%zu  scale=%.3f\n", pack->path,
                pack->tensor_count, pack->scale);
    }
    atexit(h3_lora_report);
    g_ready = 1;
    return 1;
}

static int ensure_ready(char *error, size_t error_size) {
    if (g_failed) {
        fail(error, error_size, "LoRA failed to load earlier in this process");
        return 0;
    }
    if (g_ready) return 1;
    return parse_packs(error, error_size);
}

static int pair_for_weight(h3_lora_pack *pack, const char *weight_name,
                           h3_lora_tensor **a_out, h3_lora_tensor **b_out,
                           float *alpha_out) {
    char canon[H3_LORA_NAME];
    canon_key(weight_name, canon, sizeof(canon));
    char stem[H3_LORA_NAME];
    if (!stem_before(canon, ".weight", stem, sizeof(stem))) return 0;

    char a_name[H3_LORA_NAME];
    char b_name[H3_LORA_NAME];
    static const char *a_suffix[] = {".lora_A.weight", ".lora_down.weight",
                                     ".lora.down.weight", ".lora.A.weight",
                                     NULL};
    static const char *b_suffix[] = {".lora_B.weight", ".lora_up.weight",
                                     ".lora.up.weight", ".lora.B.weight",
                                     NULL};
    h3_lora_tensor *a = NULL;
    h3_lora_tensor *b = NULL;
    for (size_t index = 0; a_suffix[index]; index++) {
        snprintf(a_name, sizeof(a_name), "%s%s", stem, a_suffix[index]);
        snprintf(b_name, sizeof(b_name), "%s%s", stem, b_suffix[index]);
        a = find_tensor(pack, a_name);
        b = find_tensor(pack, b_name);
        if (a && b) break;
        a = NULL;
        b = NULL;
    }
    if (!a || !b) return 0;
    *a_out = a;
    *b_out = b;
    *alpha_out = (float)a->shape[0];
    char alpha_name[H3_LORA_NAME];
    snprintf(alpha_name, sizeof(alpha_name), "%s.alpha", stem);
    h3_lora_tensor *alpha = find_tensor(pack, alpha_name);
    if (!alpha) {
        snprintf(alpha_name, sizeof(alpha_name), "%s.lora_alpha", stem);
        alpha = find_tensor(pack, alpha_name);
    }
    if (alpha && alpha->elements >= 1) *alpha_out = alpha->data[0];
    return 1;
}

static int gemm_delta(const h3_lora_tensor *a, const h3_lora_tensor *b,
                      uint64_t out_rows, uint64_t in_cols, float *delta) {
    /* Standard PEFT: A [rank, in], B [out, rank], W [out, in]. */
    uint64_t a0 = a->shape[0];
    uint64_t a1 = a->shape[1];
    uint64_t b0 = b->shape[0];
    uint64_t b1 = b->shape[1];
    int trans_b = 0;
    int trans_a = 0;
    uint64_t rank = 0;
    const float *B = b->data;
    const float *A = a->data;
    int ldb;
    int lda;
    if (b0 == out_rows && a1 == in_cols && b1 == a0) {
        rank = a0;
        ldb = (int)b1;
        lda = (int)a1;
    } else if (b1 == out_rows && a0 == in_cols && b0 == a1) {
        rank = a1;
        trans_b = 1;
        trans_a = 1;
        ldb = (int)b1;
        lda = (int)a1;
    } else if (b0 == out_rows && a0 == in_cols && b1 == a1) {
        rank = b1;
        trans_a = 1;
        ldb = (int)b1;
        lda = (int)a1;
    } else {
        return 0;
    }
    if (!rank || rank > INT32_MAX) return 0;
    cblas_sgemm(CblasRowMajor,
                trans_b ? CblasTrans : CblasNoTrans,
                trans_a ? CblasTrans : CblasNoTrans,
                (int)out_rows, (int)in_cols, (int)rank, 1.0f, B, ldb, A, lda,
                0.0f, delta, (int)in_cols);
    return 1;
}

int h3_lora_fuse(h3_gpu *gpu, h3_gpu_tensor *weight, const char *name,
                 uint64_t rows, uint64_t columns, char *error,
                 size_t error_size) {
    if (!ensure_ready(error, error_size)) return 0;
    if (!g_pack_count || !weight || !name || rows == 0 || columns == 0) return 1;
    int any_pair = 0;
    for (size_t pack_i = 0; pack_i < g_pack_count; pack_i++) {
        h3_lora_tensor *a = NULL;
        h3_lora_tensor *b = NULL;
        float alpha = 1.0f;
        if (pair_for_weight(&g_packs[pack_i], name, &a, &b, &alpha)) {
            any_pair = 1;
            break;
        }
    }
    if (!any_pair) {
        if (is_dit_matrix(name)) {
            for (size_t pack_i = 0; pack_i < g_pack_count; pack_i++) {
                g_packs[pack_i].missing++;
            }
        }
        return 1;
    }
    size_t elements = (size_t)rows * (size_t)columns;
    uint16_t *bf16 = malloc(elements * sizeof(uint16_t));
    float *host = malloc(elements * sizeof(float));
    float *delta = malloc(elements * sizeof(float));
    if (!bf16 || !host || !delta) {
        free(bf16);
        free(host);
        free(delta);
        fail(error, error_size, "out of memory fusing LoRA into %s", name);
        return 0;
    }
    if (!h3_gpu_tensor_read_bf16(weight, bf16, elements)) {
        free(bf16);
        free(host);
        free(delta);
        fail(error, error_size, "cannot read %s to fuse LoRA: %s", name,
             h3_gpu_error(gpu));
        return 0;
    }
    for (size_t index = 0; index < elements; index++) {
        host[index] = bf16_to_f32(bf16[index]);
    }
    int applied = 0;
    for (size_t pack_i = 0; pack_i < g_pack_count; pack_i++) {
        h3_lora_pack *pack = &g_packs[pack_i];
        h3_lora_tensor *a = NULL;
        h3_lora_tensor *b = NULL;
        float alpha = 1.0f;
        if (!pair_for_weight(pack, name, &a, &b, &alpha)) continue;
        if (!gemm_delta(a, b, rows, columns, delta)) {
            fprintf(stderr, "h3: LoRA skip %s (shape mismatch in %s)\n", name,
                    pack->path);
            continue;
        }
        float rank = a->shape[0] > 0 ? (float)a->shape[0] : 1.0f;
        if (a->shape[0] != b->shape[1] && a->shape[1] == b->shape[0] &&
            a->shape[1] > 0) {
            rank = (float)a->shape[1];
        }
        float factor = pack->scale * (rank > 0.0f ? alpha / rank : pack->scale);
        for (size_t index = 0; index < elements; index++) {
            host[index] += factor * delta[index];
        }
        applied++;
        pack->fused++;
    }
    if (applied) {
        for (size_t index = 0; index < elements; index++) {
            bf16[index] = f32_to_bf16(host[index]);
        }
        if (!h3_gpu_tensor_write_bf16(weight, bf16, elements)) {
            free(bf16);
            free(host);
            free(delta);
            fail(error, error_size, "cannot write fused LoRA into %s: %s", name,
                 h3_gpu_error(gpu));
            return 0;
        }
    }
    free(bf16);
    free(host);
    free(delta);
    return 1;
}

void h3_lora_reset(void) {
    for (size_t index = 0; index < g_pack_count; index++) {
        h3_lora_pack *pack = &g_packs[index];
        h3_st_free_header(&pack->header);
        for (size_t t = 0; t < pack->tensor_count; t++) {
            free(pack->tensors[t].name);
            free(pack->tensors[t].data);
        }
        free(pack->tensors);
    }
    memset(g_packs, 0, sizeof(g_packs));
    g_pack_count = 0;
    g_ready = 0;
    g_failed = 0;
}
