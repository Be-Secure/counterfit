

# Generated by counterfit #

from counterfit.core.targets import CFTarget

class Test9(CFTarget):
    target_name = "test9"
    data_type = "text"
    task = ""
    endpoint = ""
    input_shape = ()
    output_classes = []
    classifier = ""
    sample_input_path = ""
    X = []

    def load(self):
        self.X = []

    def predict(self, x):
        return x
