import time
import cv2
import numpy as np

from .utils import draw_disparity, CameraConfig, download_gdrive_file_model

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

drivingStereo_config = CameraConfig(0.546, 1000)

class MobileStereoNet():

	def __init__(self, model_path, camera_config=drivingStereo_config):

		self.camera_config = camera_config

		download_gdrive_file_model("1LgWEwFyu9K87eu-I-xzv29cGLj1vgLTL", model_path)

		# Initialize model
		self.model = self.initialize_model(model_path)

	def __call__(self, left_img, right_img):

		return self.estimate_disparity(left_img, right_img)

	def initialize_model(self, model_path):

		self.interpreter = Interpreter(model_path=model_path, num_threads=4)
		self.interpreter.allocate_tensors()

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()

	def estimate_disparity(self, left_img, right_img):

		# Transform images for the model
		left_input_tensor = self.prepare_input(left_img)
		right_input_tensor = self.prepare_input(right_img)

		self.disparity_map = self.inference(left_input_tensor, right_input_tensor)

		return self.disparity_map

	def get_depth(self):
		return self.camera_config.f*self.camera_config.baseline/self.disparity_map

	def prepare_input(self, img):

		self.img_height, self.img_width = img.shape[:2]

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Input values should be from -1 to 1 
		img_input = cv2.resize(img, (self.input_width,self.input_height)).astype(np.float32)
		
		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		
		img_input = ((img_input/ 255.0 - mean) / std)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)

	def inference(self, left_input_tensor, right_input_tensor):

		# Note, index 0 corresponds to the right image and inex 1 to the left image
		self.interpreter.set_tensor(self.input_details[1]['index'], left_input_tensor)
		self.interpreter.set_tensor(self.input_details[0]['index'], right_input_tensor)
		self.interpreter.invoke()

		disparity = self.interpreter.get_tensor(self.output_details[0]['index'])

		# Fix output axis order error
		# Note, the output of the model needs to be transposed
		disparity = cv2.transpose(np.squeeze(disparity))

		return disparity

	def getModel_input_details(self):

		self.input_details = self.interpreter.get_input_details()

		input_shape = self.input_details[0]['shape']
		self.input_height = input_shape[1]
		self.input_width = input_shape[2]
		self.channels = input_shape[3]

	def getModel_output_details(self):

		self.output_details = self.interpreter.get_output_details()
		output_shape = self.output_details[0]['shape']

if __name__ == '__main__':

	from imread_from_url import imread_from_url
	model_path = "../models/model_float32.tflite"

	# Initialize model
	mobile_depth_estimator = MobileStereoNet(model_path)

	# Load images
	left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
	right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

	# Estimate the depth
	disparity_map = mobile_depth_estimator(left_img, right_img)

	color_disparity = draw_disparity(disparity_map)
	color_disparity = cv2.resize(color_disparity, (left_img.shape[1],left_img.shape[0]))

	cobined_image = np.hstack((left_img, right_img, color_disparity))

	cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
	cv2.imshow("Estimated disparity", cobined_image)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

	






