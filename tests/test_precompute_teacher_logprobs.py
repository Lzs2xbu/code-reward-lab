import unittest
import sys
import types


def _install_precompute_import_stubs():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.no_grad = lambda: None
    torch.cat = lambda *args, **kwargs: None
    torch.arange = lambda *args, **kwargs: None

    torch_nn = types.ModuleType("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn_functional.log_softmax = lambda *args, **kwargs: None
    torch_nn.functional = torch_nn_functional
    torch.nn = torch_nn

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda x, *args, **kwargs: x

    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", torch_nn)
    sys.modules.setdefault("torch.nn.functional", torch_nn_functional)
    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("tqdm", tqdm_module)


_install_precompute_import_stubs()

from data.precompute_teacher_logprobs import run_tests


class PrecomputeTeacherLogprobsTest(unittest.TestCase):
    def test_run_tests_accepts_markdown_fenced_teacher_response(self):
        response = """```python
def prime_num(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```"""
        tests = [
            "assert prime_num(13)==True",
            "assert prime_num(7)==True",
            "assert prime_num(-1010)==False",
        ]

        self.assertEqual(run_tests(response, tests), (3, 3))


if __name__ == "__main__":
    unittest.main()
