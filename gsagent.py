import json
import importlib

class GSagent:
    def __init__(self, registry_path="bricks/registry.json"):
        with open(registry_path, "r") as f:
            self.registry = json.load(f)

    def list_kernels(self):
        return list(self.registry.keys())

    def describe(self, kernel):
        return self.registry.get(kernel, {})

    def run(self, kernel, **kwargs):
        if kernel not in self.registry:
            raise ValueError(f"Kernel '{kernel}' not found in registry.")

        module_name, func_name = kernel.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load {kernel}: {e}")

        # Validate input args
        expected = set(self.registry[kernel]["inputs"])
        provided = set(kwargs.keys())
        if not expected.issubset(provided):
            raise ValueError(f"Missing inputs: {expected - provided}")

        return func(**{k: kwargs[k] for k in expected})

if __name__ == "__main__":
    agent = GSagent()
    print("Available kernels:", agent.list_kernels())
    
    # Example manual execution
    try:
        result = agent.run(
            "radigen.solve",
            mixture="methyl_oleate",
            temp=60,
            oxygen=0.21,
            time=72
        )
        print("Result:", result)
    except Exception as e:
        print("Error:", e)

