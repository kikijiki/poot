// Reference generator of iq3_s_reference.tsv: decodes IQ3_S blocks with llama.cpp's own
// `dequantize_row_iq3_s`, reached through ggml's public type-traits table (`to_float`).
//
// Build (against the llama.cpp source the Nix `llama-cpp` was built from; scripts/llamacpp-dump/build.sh
// leaves the shared libraries in ~/.cache/poot/llamacpp-dump/build/bin):
//   g++ -O1 -std=c++17 iq3_s_reference.cpp -I$SRC/ggml/include \
//       -L$BIN -lggml -lggml-base -lggml-cpu -Wl,-rpath,$BIN -o iq3_s_reference
// Run: one hex string of whole 110-byte blocks on stdin; the f32 little-endian values, as hex, on stdout.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main() {
    ggml_cpu_init();
    std::string hex;
    std::cin >> hex;
    std::vector<unsigned char> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); i++) {
        bytes[i] = (unsigned char) std::stoi(hex.substr(2 * i, 2), nullptr, 16);
    }
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_IQ3_S);
    const size_t blocks = bytes.size() / traits->type_size;
    std::vector<float> out(blocks * traits->blck_size);
    traits->to_float(bytes.data(), out.data(), (int64_t) out.size());
    const unsigned char * raw = reinterpret_cast<const unsigned char *>(out.data());
    for (size_t i = 0; i < out.size() * sizeof(float); i++) {
        std::printf("%02x", raw[i]);
    }
    std::printf("\n");
    return 0;
}
