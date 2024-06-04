from counterfit.core.targets import CFTarget
import tensorflow as tf
import numpy as np
import cv2

class Bes_image_classification(CFTarget):
    target_name = "bes_image_classification"
    data_type = "image"
    task = "classification"
    endpoint = "./counterfit/targets/bes_image_classification/bes-image-classification.h5"
    img_row, img_col, channel = 28, 28, 1 
    input_shape = (img_row, img_col, channel)
    output_classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    classifier = "closed-box"
    sample_input_path = f"bes_image_classification/bes_image_classification.npz"
    X = []

    def load(self):

        input_path = self.fullpath(self.sample_input_path)
        self.data = np.load(input_path, allow_pickle=True)
        
        # Load images
        images = self.data["X"].astype(np.float32) / 255. 

        # Reshape images
        reshaped_images = []
        for image in images:
            reshaped_image = cv2.resize(image, (self.img_row, self.img_col))
            reshaped_images.append(reshaped_image)
        
        self.X = np.array(reshaped_images)

        # Load the model without the optimizer
        self.model = tf.keras.models.load_model(self.endpoint, compile=False)
        
        # Compile the model with a new optimizer
        self.model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])
        
        # Save the updated model
        self.model.save("./counterfit/targets/bes_image_classification/modified_bes-image-classification.h5")

    def predict(self, x):
        predictions = self.model.predict(x)
        prediction_probabilities = predictions.tolist()
        #print('prediction_probabilities: ', prediction_probabilities)
        return prediction_probabilities
    


