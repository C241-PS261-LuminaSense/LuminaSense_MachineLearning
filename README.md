# LuminaSense Machine Learning

## Dataset That We Used

We collected and imported more than 500 images for a Human Detection Dataset from Kaggle and other sources. The dataset was then annotated with bounding boxes and published on Roboflow. We used only one class label: "person".

## Our Models

The LuminaSense application uses the custom YOLOv8 model for object detection from Ultralytics. YOLOv8 employs a convolutional neural network (CNN) architecture optimized for real-time object detection. This architecture is designed for single-stage detection, meaning that the prediction of bounding boxes and object classes is done in a single inference. Here are the details of the model steps:

1. Upload dataset to Google Drive
2. Mount Google Colab at Google Drive
3. Import dataset from drive
4. Install necessary GPU (Tesla T4 or Nvidia CUDA) to speed up image processing
5. Define the YOLOv8 model for training
6. Start training using model and dataset (it takes 20 minutes), then get a trained model (`best.pt` and `last.pt`)
7. Plot loss and accuracy from trained model
8. Test model with image and video examples
9. Convert model into TFLite or TFjs format (still failed :’)

## The Training Results

Here are the results of training the YOLOv8 model trained with CUDA version 12.1.

![GAMBAR](https://github.com/C241-PS261-LuminaSense/.github/blob/be5e91f47077610f77bccb3205489e4bf23aa1a3/assets/image1.jpg)
![GAMBAR](https://github.com/C241-PS261-LuminaSense/.github/blob/be5e91f47077610f77bccb3205489e4bf23aa1a3/assets/image2.jpg)

### Performance Metrics
1. **Precision 0.916**: Precision indicates the percentage of positive detections that are truly positive. In this case, 91.6% of the boxes drawn by the model as objects are indeed the detected objects (true positives). This value shows that the model rarely makes false positive detections (marking "person" when it's not actually a "person").
2. **Recall 0.852**: Recall is also quite high at 85.2%. This indicates that the model successfully detects most of the objects present in the images and only misses a few objects (false negatives).
3. **mAP50 0.923**: mAP50 (mean Average Precision) is the average precision of the model at an IoU threshold of 0.5 on the precision-recall curve. A value of 0.923 shows that the model has a high level of accuracy in detecting objects in images overall.
4. **mAP50-95 0.717**: mAP50-95 (mean Average Precision) is the average precision of the model across the range of 50% to 95% IoU thresholds on the precision-recall curve. A value of 0.717 indicates that the model has lower precision in detecting objects at lower confidence levels. This could be due to factors such as small objects, obstructed objects, or objects with poor lighting.

### Example of the model's testing results on a photo:
![GAMBAR](https://github.com/C241-PS261-LuminaSense/.github/blob/be5e91f47077610f77bccb3205489e4bf23aa1a3/assets/image3.jpg)

### Example of the model's testing results on a video:
![GIF](https://user-images.githubusercontent.com/37643248/188248210-2c02790b-6231-4549-8211-e3edcccba9e8.gif)
