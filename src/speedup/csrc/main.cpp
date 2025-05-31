#include <torch/extension.h>
#include "ops/cutlass/cutlass_binding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    speedup::init_cutlass_binding(m);
}