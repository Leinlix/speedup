import torch
import speedup.C

def run_test():
    x = torch.randint(32, (1, 8), dtype=torch.uint8)
    print(x)
    y = speedup._C.cutlass_test_func(x)
    print(y)

if __name__ == '__main__':
    run_test()
