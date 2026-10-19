// llamacpp-dequant-rows: dequantize single rows of a GGUF's tensors with llama.cpp's own `to_float` for the
// tensor's type, and print per row `name type row v0 v1 v2 v3 sum abs-sum weighted-sum` (tab separated, f64):
// the first four values, the sum, the sum of absolute values and the position-weighted sum `sum (i + 1) * v_i`
// (which fails a decoder that permutes values within a block). The reference side of the per-row dequant rows in
// crates/poot-models/src/architectures/qwen38_27b.rs. Built by build.sh against the pinned source.
//
//   llamacpp-dequant-rows <model.gguf> <tensor:row>...
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tensor:row>...\n", argv[0]);
        return 2;
    }
    struct ggml_context * meta = nullptr;
    gguf_init_params params = { /*no_alloc=*/true, &meta };
    struct gguf_context * gguf = gguf_init_from_file(argv[1], params);
    if (gguf == nullptr) {
        return 2;
    }
    FILE * file = std::fopen(argv[1], "rb");
    if (file == nullptr) {
        return 2;
    }
    const size_t data = gguf_get_data_offset(gguf);
    for (int a = 2; a < argc; a++) {
        const std::string arg = argv[a];
        const size_t colon = arg.rfind(':');
        const std::string name = arg.substr(0, colon);
        const int64_t row = std::atoll(arg.c_str() + colon + 1);
        const int64_t id = gguf_find_tensor(gguf, name.c_str());
        if (colon == std::string::npos || id < 0) {
            std::fprintf(stderr, "no tensor %s\n", arg.c_str());
            return 2;
        }
        const struct ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
        std::vector<uint8_t> raw(row_bytes);
        std::vector<float> out(t->ne[0]);
        const size_t at = data + gguf_get_tensor_offset(gguf, id) + (size_t) row * row_bytes;
        if (row >= t->ne[1] || std::fseek(file, (long) at, SEEK_SET) != 0 ||
            std::fread(raw.data(), 1, row_bytes, file) != row_bytes) {
            std::fprintf(stderr, "cannot read %s\n", arg.c_str());
            return 2;
        }
        ggml_get_type_traits(t->type)->to_float(raw.data(), out.data(), t->ne[0]);
        double sum = 0, abs_sum = 0, weighted = 0;
        for (size_t i = 0; i < out.size(); i++) {
            sum += out[i];
            abs_sum += std::fabs(out[i]);
            weighted += (double) (i + 1) * out[i];
        }
        std::printf("%s\t%d\t%lld\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n", name.c_str(), (int) t->type,
                    (long long) row, (double) out[0], (double) out[1], (double) out[2], (double) out[3], sum,
                    abs_sum, weighted);
    }
    return 0;
}
