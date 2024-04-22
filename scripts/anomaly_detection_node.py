#!/usr/bin/env python
import rospy
import roslib
import rospkg
import os
import numpy as np
import cv2
from decision_msgs.msg import Decision
from sensor_msgs.msg import CompressedImage
from ..config import cfg
from anomaly_detector import AnomalyDetector
roslib.load_manifest('anomaly_detection')


class AnomalyNode(AnomalyDetector):

    def __init__(self, base_dir, cfg):
        super().__init__(base_dir, cfg)
        rate = float(rospy.get_param('~rate', '1.0'))
        anomaly_topic = rospy.get_param('~anomaly_topic', 'anomaly')
        image_topic = rospy.get_param('~image_topic', 'image')
        rospy.loginfo('[AnomalyDetection] rate = %d', rate)
        rospy.loginfo('[AnomalyDetection] out_topic = %s', anomaly_topic)
        self.current_image=None
        pub = rospy.Publisher(anomaly_topic, Decision, queue_size=10)

        def img_callback(msg):
            np_arr = np.fromstring(msg.data, np.uint8)
            self.current_image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)

        image_sub = rospy.Subscriber(
            image_topic, CompressedImage, callback=img_callback, queue_size=10)
        
        msg = Decision()
        msg.target = 'ANOMALY'
        msg.confidence = 0.0

        while not rospy.is_shutdown():
            anomaly_score = self.getAnomalyScore(self.current_image)
            msg.confidence = anomaly_score
            pub.publish(msg)
            if rate:
                rospy.sleep(1.0/rate)
            else:
                rospy.sleep(1.0)


if __name__ == '__main__':
    rospy.init_node('AnomalyDetection')
    rospack = rospkg.RosPack()
    base_dir = rospack.get_path('anomaly_detection')
    cfg_file = os.path.join(base_dir, 'config/ae_predictor_config.yaml')
    cfg.merge_from_file(cfg_file)
    try:
        dn = AnomalyNode(base_dir, cfg)
    except rospy.ROSInterruptException:
        pass
