#include <torch/extension.h>

namespace speedup {
    void init_cutlass_binding(py::module &m);
}