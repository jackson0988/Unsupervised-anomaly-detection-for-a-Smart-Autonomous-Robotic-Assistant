# Unsupervised anomaly detection for a Smart Autonomous Robotic Assistant Surgeon (SARAS)using a deep residual autoencoder
Anomaly detection in Minimally-Invasive Surgery (MIS) traditionally requires a human expert monitoring the procedure from a console. Data scarcity, on the other hand, hinders what would be a desirable migration towards autonomous robotic-assisted surgical systems. Automated anomaly detection systems in this area typically rely on classical supervised learning. Anomalous events in a surgical setting, however, are rare, making it difficult to capture data to train a detection model in a supervised fashion. In this work we thus propose an unsupervised approach to anomaly detection for robotic-assisted surgery based on deep residual autoencoders. The idea is to make the autoencoder learn the 'normal' distribution of the data and detect abnormal events deviating from this distribution by measuring the reconstruction error. The model is trained and validated upon both the publicly available Cholec80 dataset, provided with extra annotation, and on a set of videos captured on procedures using artificial anatomies ('phantoms') produced as part of the Smart Autonomous Robotic Assistant Surgeon (SARAS) project. The system achieves recall and precision equal to 78.4%, 91.5%, respectively, on Cholec80 and of 95.6%, 88.1% on the SARAS phantom dataset. The end-to-end system was developed and deployed as part of the SARAS demonstration platform for real-time anomaly detection with a processing time of about 25 ms per frame. 

Pre-requisites:

Skills: Some familiarity with concepts and frameworks of neural networks:

    Framework: Keras and Tensorflow
    Concepts: convolutional, Residual Autoencoder.
    Moderate skills in coding with Python and machine learning using Python.

Software Dependencies: Various python modules. We recommend working with a conda environement Python 3.7 PIL glob cv2 numpy matplotlib 3.2.2 sklearn CUDA Toolkit 10.1 cudnn 7.6.5 TensorFlow 2.2.0