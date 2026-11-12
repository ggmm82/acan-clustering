import contextlib
import io
import json
import unittest

from scripts import verify_math_foundation


class MathFoundationTest(unittest.TestCase):
    def test_documented_properties_hold_on_the_production_operator(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            verify_math_foundation.main()      # every documented claim is an assertion inside
        report = json.loads(output.getvalue())
        self.assertTrue(report['same_partition'])
        self.assertTrue(report['clustered_attention']['bound_holds'])
        self.assertLessEqual(report['kernel_rank'], 8)


if __name__ == '__main__':
    unittest.main()
