import os
import sys
import cv2
import torch
from anomaly_detector import AnomalyDetector
from defaults import _C as cfg


class detectorNode(AnomalyDetector):

	def __init__(self, base_dir, cfg):
		super().__init__(base_dir, cfg)
		self.current_image=None	
		#self.compute()

	def compute(self):
		#load video with cv2
		vidObj = cv2.VideoCapture(sys.argv[2])
		# Test anomaly score for video with n frames
		n=4800
		for i in range(n):
			success, image=vidObj.read()
			print(image.shape)
			if success:
				anomaly_score = self.getAnomalyScore(image)
				print(anomaly_score)
			else:
				print("Fail")


def main_test(argv):
	base_dir=argv[1]
	cfg_file = os.path.join(base_dir, 'config/ae_predictor_config.yaml')
	cfg.merge_from_file(cfg_file)
	detector=detectorNode(base_dir, cfg)
	if not os.path.isdir(os.path.join(base_dir, cfg.DIR.saved_models, 'traced')):
	    os.mkdir(os.path.join(base_dir, cfg.DIR.saved_models, 'traced'))
	trace_path = os.path.join(base_dir, cfg.DIR.saved_models, 'traced', cfg.TEST.enc_dec_model)
	# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
	#traced_script_module = torch.jit.trace(model, input)
	sm = torch.jit.script(detector.model)
	sm.save(trace_path)


if __name__=="__main__":
	main_test(sys.argv)



