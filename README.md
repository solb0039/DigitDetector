
#### Required project organization for relative paths to work:
```
+--Digit_Detector
|   +--images
|       +--test
|       +--train
|   +--graded_images
|       +--unannotated_images
|           +--1.png
|           +--2.png
|           +--3.png
|           +--4.png
|           +--5.png
|   +--cnn_models
|       +--vgg16_model.py
|       +--optimized_vgg16_net.pth
|   +--CustomDataLoader.py
|   +--DigitDetector.py
|   +--RegionsOfInterest.py
|   +--run.py
|   +--env.yml
```

#### Environment Set-up
```
$ conda env create --name digit_project --file=env.yml
$ conda activate digit_project
```

#### To run basic image classification on graded images:
```
$ python run.py
```

#### To train VGG16 or custom CNN model
-Download and extract training and test images in `images` directory
-Edit the hyperparameter settings in either `vgg_model.py` or `custom_model.py` files
```
$ python ./image_processing/read_data.py
$ python ./cnn_models/vgg_model.py
```
