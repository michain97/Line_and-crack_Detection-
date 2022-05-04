# Line_and-crack_Detection-
## Project for Deep Learning Class

### Introduction:

A deep understanding of lane boundary position is vital for vehicle safety, which is needed to avoid collisions and localize vehicle. Lanes boundaries detection accuracy has increased significantly since introduction of deep learning, with majority of recent systems using convolutional neural networks to process sensorial data and infer high-level information. The use of LiDAR is extremely expensive therefore as an alternative, comparatively cheaper cameras are used which can utilize chromatic differences on the road.

### Aim:

#### Lane Detection:
We trained the CNN to recognize lane boundaries, rather than lane markings to avoid grouping different lane markings in a lane boundary to save processing time. 
Instead of semantic segmentation, we performed instance segmentation on lane boundaries, to distinguish different lane boundaries without relying on clustering algorithms.
We used  ERFNet architecture as baseline model since Mask R-CNN is not suggested for use in real-time application

![image](https://user-images.githubusercontent.com/104802856/166630316-ecba14f2-673a-47b2-b728-da7a6dc87ca4.png)


#### Crack Detection:
To detect crack, we used classification approach, in which, images and label are fed to deep learning architecture (50 layers deep resnet50). The model learns from it and when unknown images are provided, it predicts the label.

![image](https://user-images.githubusercontent.com/104802856/166630379-f9823e8c-ebba-48c7-8729-20baa90c967a.png)

### Dataset: 
TuSimple dataset is used for lane detection. It is consists of 6408 1280×720 road images on US highways, divided in 3626 for training, and 2782 for testing. 410 images extracted from the training set have been used as validation set during training.
![image](https://user-images.githubusercontent.com/104802856/166630466-910904fd-0f2e-41d7-bda2-12affe4cbc5c.png)


### Results: 
#### Lane Detection 

![image](https://user-images.githubusercontent.com/104802856/166630719-493dbd84-84cd-4d17-92e4-6f2cf44a2486.png)
![image](https://user-images.githubusercontent.com/104802856/166630733-9b029bac-ce61-46eb-a7ec-0d9e7a2ef2bf.png)

#### Crack Detection 

The table below tracks the accuracy at which cracks are identified. It tracks False Positives (FP), True Positives (TP) and False Negatives (FN), as well a True Negatives (TN) 


###### Epoch 5

![image](https://user-images.githubusercontent.com/104802856/166630857-4cc64367-7ddd-4504-8351-532226725b8a.png)
![image](https://user-images.githubusercontent.com/104802856/166630861-e152eea6-677c-4eb6-b589-98139a665ab9.png)

###### Epoch 10

![image](https://user-images.githubusercontent.com/104802856/166630899-4a0a80c3-9b6b-4a18-8aed-e5b5fe1fad72.png)
![image](https://user-images.githubusercontent.com/104802856/166630906-e7d76053-9413-477d-89a3-b9d84986d39e.png)

###### Epoch 15 

![image](https://user-images.githubusercontent.com/104802856/166630928-70581849-a267-4b3f-ac88-8a97f3a2b430.png)
![image](https://user-images.githubusercontent.com/104802856/166630945-9a42c1c1-1bc9-4bd7-831a-620cd91caf07.png)

### Conclusion 
We have successfully implemented Lane detection and further added crack detection feature in it (with the accuracy of 88% and F1 score 0.90).
The number of FN and FP is high because we used 10 epochs (a hyperparameter that defines the number of times that the algorithm will work through the entire training dataset). This impacts the precision of the algorithm as TP/(TP+FP). The graphs bellow shows the increase of the precision as the epoch increase. It also depends on the size of the architecture i.e., a much more advanced CNN.
