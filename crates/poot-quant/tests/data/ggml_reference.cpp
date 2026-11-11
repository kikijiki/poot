// Reference generator of tests/data/ggml_reference/*.tsv: decodes whole blocks of any ggml type with
// llama.cpp's own dequantizer, reached through ggml's public type-traits table (`to_float`). One program for
// every format: the ggml type id is the only argument.
//
// Build (against the llama.cpp source the Nix `llama-cpp` was built from; scripts/llamacpp-dump/build.sh
// leaves the shared libraries in ~/.cache/poot/llamacpp-dump/build/bin):
//   g++ -O1 -std=c++17 ggml_reference.cpp -I$SRC/ggml/include \
//       -L$BIN -lggml -lggml-base -lggml-cpu -Wl,-rpath,$BIN -o ggml_reference
// Run: `ggml_reference <ggml type id>` with one hex string of whole blocks on stdin; the f32 little-endian
// values, as hex, on stdout. A type without a `to_float` (Q8_1, Q8_K) is refused.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ggml_reference <ggml type id>\n");
        return 2;
    }
    const auto type = static_cast<ggml_type>(std::atoi(argv[1]));
    ggml_cpu_init();
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits->to_float == nullptr) {
        std::fprintf(stderr, "ggml type %d (%s) has no to_float\n", (int) type, traits->type_name);
        return 1;
    }
    std::string hex;
    std::cin >> hex;
    std::vector<unsigned char> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); i++) {
        bytes[i] = (unsigned char) std::stoi(hex.substr(2 * i, 2), nullptr, 16);
    }
    if (bytes.size() % traits->type_size != 0) {
        std::fprintf(stderr, "%zu bytes are not whole %zu-byte blocks\n", bytes.size(), traits->type_size);
        return 1;
    }
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
