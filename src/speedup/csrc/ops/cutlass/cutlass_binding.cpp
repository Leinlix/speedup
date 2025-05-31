#include "cutlass_binding.h"
#include "cutlass_test_func.h"

namespace speedup {
    void init_cutlass_binding(py::module &m){
        m.def("cutlass_test_func", cutlass_test_func);
    }
}