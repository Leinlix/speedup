
#include <torch/extension.h>

namespace speedup{
    torch::Tensor cutlass_test_func(torch::Tensor &input);
}