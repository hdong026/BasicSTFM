import unittest

from basicstfm.registry import Registry


class RegistryTest(unittest.TestCase):
    def test_build_merges_params_and_direct_keys(self):
        registry = Registry("toy")

        @registry.register()
        class Toy:
            def __init__(self, a, b=0, c=0):
                self.a = a
                self.b = b
                self.c = c

        obj = registry.build({"type": "Toy", "params": {"a": 1, "b": 2}, "b": 3}, c=4)
        self.assertEqual(obj.a, 1)
        self.assertEqual(obj.b, 3)
        self.assertEqual(obj.c, 4)

    def test_unknown_name_reports_available_items(self):
        registry = Registry("toy")
        with self.assertRaises(KeyError) as ctx:
            registry.build({"type": "Missing"})
        self.assertIn("Available", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
