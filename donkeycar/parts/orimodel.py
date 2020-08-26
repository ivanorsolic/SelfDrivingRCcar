
from donkeycar.parts.keras import KerasPilot

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.python.keras.layers.merge import Concatenate

import cv2
import numpy as np

class OriModel(KerasPilot):
    '''
    Custom model that takes an input image and feeds it and a preprocessed version of it to the model.
    The preprocessing converts the image to HSL color space, extracts the S channel and thresholds it.
    The thresholded S channel is passed to the model to help find lane lines easier.
    '''
    def __init__(self, model=None, input_shape=(180, 320, 3), num_behavior_inputs=2, *args, **kwargs):
        super(OriModel, self).__init__(*args, **kwargs)
        self.model = oriModel(inputShape=input_shape, numberOfBehaviourInputs = num_behavior_inputs)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                loss='mse')

    def run(self, inputImage, behaviourArray):
        # Preprocesses the input image for easier lane detection
        extractedLaneInput = self.processImage(inputImage)
        # Reshapes to (1, height, width, channels)
        extractedLaneInput = extractedLaneInput.reshape((1,) + extractedLaneInput.shape)
        inputImage = inputImage.reshape((1,) + inputImage.shape)

        behaviourArray = np.array(behaviourArray).reshape(1,len(behaviourArray))
        # Predicts the output steering and throttle
        steering, throttle = self.model.predict([inputImage, extractedLaneInput, behaviourArray])
        #print("Throttle: %f, Steering: %f" % (throttle[0][0], steering[0][0]))
        return steering[0][0], throttle[0][0]
        
    def warpImage(self, image):
        # Define the region of the image we're interested in transforming
        regionOfInterest = np.float32(
                        [[0,  180],  # Bottom left
                        [112.5, 87.5], # Top left
                        [200, 87.5], # Top right
                        [307.5, 180]]) # Bottom right
                    
        # Define the destination coordinates for the perspective transform
        newPerspective = np.float32(
                        [[80,  180],  # Bottom left
                        [80,    0.25],  # Top left
                        [230,   0.25],  # Top right
                        [230, 180]]) # Bottom right
        # Compute the matrix that transforms the perspective
        transformMatrix = cv2.getPerspectiveTransform(regionOfInterest, newPerspective)
        # Warp the perspective - image.shape[:2] takes the height, width, [::-1] inverses it to width, height
        warpedImage = cv2.warpPerspective(image, transformMatrix, image.shape[:2][::-1], flags=cv2.INTER_LINEAR)
        return warpedImage
        
    def extractLaneLinesFromSChannel(self, warpedImage):
        # Convert to HSL
        hslImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2HLS)
        # Split the image into three variables by the channels
        hChannel, lChannel, sChannel = cv2.split(hslImage)
        # Threshold the S channel image to select only the lines
        lowerThreshold = 65
        higherThreshold = 255
        # Threshold the image, keeping only the pixels/values that are between lower and higher threshold
        returnValue, binaryThresholdedImage = cv2.threshold(sChannel,lowerThreshold,higherThreshold,cv2.THRESH_BINARY)
        # Since this is a binary image, we'll convert it to a 3-channel image so OpenCV can use it
        thresholdedImage = cv2.cvtColor(binaryThresholdedImage, cv2.COLOR_GRAY2RGB)
        return thresholdedImage

    def processImage(self, image):
        one_byte_scale = 1.0 / 255.0 
        warpedImage = self.warpImage(image)
        warpedImage = cv2.normalize(warpedImage,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        thresholdedImage = self.extractLaneLinesFromSChannel(warpedImage)
        return np.array(thresholdedImage).astype(np.float32) * one_byte_scale

def oriModel(inputShape, numberOfBehaviourInputs):

    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob
    
    # Input layers
    imageInput = Input(shape=inputShape, name='imageInput')
    laneInput = Input(shape=inputShape, name='laneInput')
    behaviourInput = Input(shape=(numberOfBehaviourInputs,), name="behaviourInput")

    # Input image convnet
    x = imageInput
    x = Conv2D(24, (5,5), strides=(2,2), name="Conv2D_imageInput_1")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(32, (5,5), strides=(2,2), name="Conv2D_imageInput_2")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (5,5), strides=(2,2), name="Conv2D_imageInput_3")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_imageInput_4")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_imageInput_5")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Flatten(name="flattenedx")(x)
    x = Dense(100)(x)
    x = Dropout(rate)(x)

    # Preprocessed lane image input convnet
    y = laneInput
    y = Conv2D(24, (5,5), strides=(2,2), name="Conv2D_laneInput_1")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(32, (5,5), strides=(2,2), name="Conv2D_laneInput_2")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (5,5), strides=(2,2), name="Conv2D_laneInput_3")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_4")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_5")(y)
    y = LeakyReLU()(y)
    y = Flatten(name="flattenedy")(y)
    y = Dense(100)(y)
    y = Dropout(rate)(y)

    # Behavioural net
    z = behaviourInput
    z = Dense(numberOfBehaviourInputs * 2, activation='relu')(z)
    z = Dense(numberOfBehaviourInputs * 2, activation='relu')(z)
    z = Dense(numberOfBehaviourInputs * 2, activation='relu')(z)

    # Concatenated final convnet
    c = Concatenate(axis=1)([x, y])
    c = Dense(100, activation='relu')(c)
    
    o = Concatenate(axis=1)([z, c])
    o = Dense(100, activation='relu')(o)
    o = Dense(50, activation='relu')(o)
    
    # Output layers
    steering_out = Dense(1, activation='linear', name='steering_out')(o)
    throttle_out = Dense(1, activation='linear', name='throttle_out')(o)
    model = Model(inputs=[imageInput, laneInput, behaviourInput], outputs=[steering_out, throttle_out]) 
    
    return model