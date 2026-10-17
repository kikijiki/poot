// llamacpp-dump: decode a token list with llama.cpp on the CPU and write every f32/f16/bf16/i32 graph tensor
// it computes to a directory, one `<seq>.f32` file per tensor (logical row-major order, widened to f32) and
// an `index.tsv` of `seq name op ne0 ne1 ne2 ne3` lines. The reference side of the POOT-1156 per-layer bisect
// (crates/poot-llm/src/driver/llamacpp_bisect_tests.rs). Built by build.sh against the Nix-pinned source.
//
//   llamacpp-dump <model.gguf> <out-dir> <tokens.txt>
//
// `tokens.txt` holds whitespace-separated token ids; they are decoded as one batch from position 0 with
// logits for every position, in a context exactly as large as the batch.
#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct Sink {
    std::string dir;
    int seq = 0;
    std::vector<uint8_t> raw;
    FILE * index = nullptr;
};

static bool on_tensor(struct ggml_tensor * t, bool ask, void * user) {
    auto * sink = static_cast<Sink *>(user);
    if (ask) {
        return true;
    }
    if (ggml_is_quantized(t->type)) {
        return true;
    }
    const size_t bytes = ggml_nbytes(t);
    sink->raw.resize(bytes);
    ggml_backend_tensor_get(t, sink->raw.data(), 0, bytes);
    std::vector<float> out(ggml_nelements(t));
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const uint8_t * p = sink->raw.data() + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                    switch (t->type) {
                        case GGML_TYPE_F32:  out[k++] = *reinterpret_cast<const float *>(p); break;
                        case GGML_TYPE_F16:  out[k++] = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p)); break;
                        case GGML_TYPE_BF16: out[k++] = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(p)); break;
                        case GGML_TYPE_I32:  out[k++] = static_cast<float>(*reinterpret_cast<const int32_t *>(p)); break;
                        default: return true;
                    }
                }
            }
        }
    }
    const std::string path = sink->dir + "/" + std::to_string(sink->seq) + ".f32";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    std::fprintf(sink->index, "%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\n", sink->seq, t->name, ggml_op_name(t->op),
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    sink->seq++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <out-dir> <tokens.txt>\n", argv[0]);
        return 2;
    }
    std::vector<llama_token> tokens;
    {
        std::ifstream in(argv[3]);
        int id;
        while (in >> id) {
            tokens.push_back(id);
        }
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "no tokens in %s\n", argv[3]);
        return 2;
    }

    Sink sink;
    sink.dir = argv[2];
    sink.index = std::fopen((sink.dir + "/index.tsv").c_str(), "w");
    if (sink.index == nullptr) {
        std::fprintf(stderr, "cannot write %s/index.tsv\n", argv[2]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        return 2;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = tokens.size();
    cparams.n_batch = tokens.size();
    cparams.n_ubatch = tokens.size();
    cparams.cb_eval = on_tensor;
    cparams.cb_eval_user_data = &sink;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        return 2;
    }
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = tokens.size();
    const int status = llama_decode(ctx, batch);
    std::fclose(sink.index);
    if (status != 0) {
        std::fprintf(stderr, "llama_decode failed: %d\n", status);
        return 1;
    }
    return 0;
}
