# Line_and-crack_Detection-
Project for Deep Learning Class

Introduction -

A deep understanding of lane boundary position is vital for vehicle safety, which is needed to avoid collisions and localize vehicle. Lanes boundaries detection accuracy has increased significantly since introduction of deep learning, with majority of recent systems using convolutional neural networks to process sensorial data and infer high-level information. The use of LiDAR is extremely expensive therefore as an alternative, comparatively cheaper cameras are used which can utilize chromatic differences on the road.

Aim! 

Lane Detection
We trained the CNN to recognize lane boundaries, rather than lane markings to avoid grouping different lane markings in a lane boundary to save processing time. 
Instead of semantic segmentation, we performed instance segmentation on lane boundaries, to distinguish different lane boundaries without relying on clustering algorithms.
We used  ERFNet architecture as baseline model since Mask R-CNN is not suggested for use in real-time application

![image](https://user-images.githubusercontent.com/104802856/166630316-ecba14f2-673a-47b2-b728-da7a6dc87ca4.png)


Crack Detection
To detect crack, we used classification approach, in which, images and label are fed to deep learning architecture (50 layers deep resnet50). The model learns from it and when unknown images are provided, it predicts the label.

![image](https://user-images.githubusercontent.com/104802856/166630379-f9823e8c-ebba-48c7-8729-20baa90c967a.png)

Dataset 
TuSimple dataset is used for lane detection. It is consists of 6408 1280×720 road images on US highways, divided in 3626 for training, and 2782 for testing. 410 images extracted from the training set have been used as validation set during training.
![image](https://user-images.githubusercontent.com/104802856/166630466-910904fd-0f2e-41d7-bda2-12affe4cbc5c.png)


Results 


