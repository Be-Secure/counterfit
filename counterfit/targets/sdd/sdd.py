

# Generated by counterfit #

from counterfit.core.targets import Target

class Sdd(Target):
    target_name = "sdd"
    target_data_type = "image"
    target_endpoint = ""
    target_input_shape = ()
    target_output_classes = []
    target_classifier = ""
    X = []

    def load(self):
        self.X = []

    def predict(self, x):
        return x
