// llamacpp-truncate: copy a GGUF keeping its metadata, its model-level tensors (everything not under
// `blk.`) and only the first N blocks, with `<arch>.block_count` set to N and the multi-token-prediction
// count dropped, so llama.cpp loads and runs a real model's first layers from a few GB instead of the
// whole file. Tensor bytes are copied verbatim (still quantized), except that the quantized tensors of block
// `f32-block` are dequantized to f32 with llama.cpp's own `to_float`, so llama.cpp multiplies that block's
// activations in f32 (a quantized weight makes it quantize the activations too, which an f32 reference does not
// do). Built by build.sh against the pinned source.
//
//   llamacpp-truncate <in.gguf> <out.gguf> <blocks> [f32-block]
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool keep(const char * name, int blocks) {
    if (std::strncmp(name, "blk.", 4) != 0) {
        return true;
    }
    return std::atoi(name + 4) < blocks;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <blocks> [f32-block]\n", argv[0]);
        return 2;
    }
    const int blocks = std::atoi(argv[3]);
    const int f32_block = argc == 5 ? std::atoi(argv[4]) : -1;
    auto dequantized = [&](const char * name, enum ggml_type type) {
        return std::strncmp(name, "blk.", 4) == 0 && std::atoi(name + 4) == f32_block && ggml_is_quantized(type);
    };
    struct ggml_context * meta = nullptr;
    gguf_init_params params = { /*no_alloc=*/true, &meta };
    struct gguf_context * src = gguf_init_from_file(argv[1], params);
    if (src == nullptr) {
        return 2;
    }
    const int64_t arch_key = gguf_find_key(src, "general.architecture");
    const std::string arch = gguf_get_val_str(src, arch_key);

    struct gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    gguf_set_val_u32(dst, (arch + ".block_count").c_str(), (uint32_t) blocks);
    gguf_remove_key(dst, (arch + ".nextn_predict_layers").c_str());

    std::vector<int64_t> kept;
    for (int64_t i = 0; i < gguf_get_n_tensors(src); i++) {
        const char * name = gguf_get_tensor_name(src, i);
        if (keep(name, blocks)) {
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            gguf_add_tensor(dst, t);
            if (dequantized(name, t->type)) {
                gguf_set_tensor_type(dst, name, GGML_TYPE_F32);
            }
            kept.push_back(i);
        }
    }
    if (!gguf_write_to_file(dst, argv[2], /*only_meta=*/true)) {
        return 2;
    }

    FILE * in = std::fopen(argv[1], "rb");
    FILE * out = std::fopen(argv[2], "ab");
    if (in == nullptr || out == nullptr) {
        return 2;
    }
    const size_t src_data = gguf_get_data_offset(src);
    size_t written = 0;
    std::vector<uint8_t> buf;
    for (size_t k = 0; k < kept.size(); k++) {
        const int64_t i = kept[k];
        const size_t want = gguf_get_tensor_offset(dst, (int64_t) k);
        for (; written < want; written++) {
            std::fputc(0, out);
        }
        const char * name = gguf_get_tensor_name(src, i);
        const size_t size = gguf_get_tensor_size(src, i);
        buf.resize(size);
        if (std::fseek(in, (long) (src_data + gguf_get_tensor_offset(src, i)), SEEK_SET) != 0 ||
            std::fread(buf.data(), 1, size, in) != size) {
            std::fprintf(stderr, "read of %s failed\n", name);
            return 2;
        }
        const enum ggml_type type = gguf_get_tensor_type(src, i);
        if (dequantized(name, type)) {
            const struct ggml_tensor * t = ggml_get_tensor(meta, name);
            std::vector<float> wide(ggml_nelements(t));
            ggml_get_type_traits(type)->to_float(buf.data(), wide.data(), (int64_t) wide.size());
            if (std::fwrite(wide.data(), sizeof(float), wide.size(), out) != wide.size()) {
                return 2;
            }
            written += wide.size() * sizeof(float);
        } else {
            if (std::fwrite(buf.data(), 1, size, out) != size) {
                return 2;
            }
            written += size;
        }
    }
    std::fclose(in);
    std::fclose(out);
    return 0;
}
