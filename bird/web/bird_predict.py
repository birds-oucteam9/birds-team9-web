import numpy as np
#Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

MODEL_PATH = 'birds.h5'

# Load your trained model
model = load_model(MODEL_PATH,compile=False)
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224)) #Change to 224 224

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    #preds = model.predict(x)
    #preds = model.predict_classes(x)
    preds = np.argmax(model.predict(x), axis=-1)
    #preds=(model.predict(x) > 0.5).astype("int32")

    return preds
