# Prerequistes

Install the required packages to execute our code with **pip** by running:

```
pip install -r requirements.txt
```

# The Application
We have trained different models for Emotion Recognition from human facial Expressions. They can be used either to perform live classification for the webcam input (webcam_demo.py) or to classify a picture that can be chosen with the file manager (classify_picture.py). 

### Webcam Demo
- Start the Webcam Demo with ```python webcam_demo.py```
- Press ```q``` to close the webcam window and terminate the application
### Classify Picture 
Start the program with ```python classify_picture.py```. You can chose a picture with the file manager. The classified picture will be displayed with your default image viewer and you can save it with your file manager.

# Models

Our trained models are stored in ```saved_models```:

- ```deepCNN.h5```: first trained model with 7 classes
- ```improved_deepCNN_1.h5```: model trained without disgusted class
- ```improved_deepCNN_2.h5```: model trained without disgusted class and without FER-2013 dataset


You can choose a different model for the Webcam Demo Application by changing the filepath in line 27 in webcam_demo.py

