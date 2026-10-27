# Technical Design Document

## 1. Project Title
AI-Powered Sign Language to Speech Translator for Inclusive Communication

## 2. Purpose
The project demonstrates a computer-vision prototype that recognizes a small set of user-trained static hand gestures and converts the predicted labels into text and optional speech.

## 3. System Architecture

    User
      |
      v
    Streamlit User Interface
      |
      v
    Camera Image Capture
      |
      v
    OpenCV Image Processing
      |
      v
    MediaPipe Hand Landmark Detection
      |
      v
    Landmark Normalization
      |
      +----------------------+
      |                      |
      v                      v
    Training Mode         Prediction Mode
      |                      |
      v                      v
    CSV Sample Store      Trained KNN Classifier
      |                      |
      v                      v
    Model Training        Predicted Gesture Label
                             |
                             v
                      Text Display
                             |
                             v
                      Optional Speech Output

## 4. Technology Stack

- Python: application programming
- Streamlit: user interface
- OpenCV: image decoding and color conversion
- MediaPipe: hand landmark detection
- NumPy: numerical processing
- Pandas: sample-data management
- Scikit-learn: feature scaling and K-nearest-neighbours classification
- pyttsx3: optional text-to-speech output

## 5. Main Components

### 5.1 User Interface
Streamlit provides camera input, gesture-label entry, sample collection, model training, and prediction display.

### 5.2 Image Processing
OpenCV decodes captured images and converts them to RGB format for hand detection.

### 5.3 Hand Landmark Detection
MediaPipe identifies 21 landmarks on one visible hand. Each landmark provides an x and y coordinate.

### 5.4 Feature Normalization
The application uses 42 coordinate values. Coordinates are shifted relative to the wrist and scaled to reduce differences in hand position and size.

### 5.5 Training and Prediction
The user collects labeled samples. A K-nearest-neighbours classifier is trained on these features and predicts labels for new captured gestures.

### 5.6 Text and Speech
The predicted label appears on screen. The user can request speech output through pyttsx3 when supported by the system.

## 6. Data Flow

1. The user enters a gesture label.
2. The camera captures an example.
3. MediaPipe extracts hand landmarks.
4. The application normalizes the coordinates.
5. The labeled features are stored in a local CSV file.
6. The classifier learns from samples for at least two labels.
7. A new image is processed using the same steps.
8. The model predicts a label, which is displayed as text.
9. Optional text-to-speech reads the predicted label.

## 7. Data Storage
Training samples are stored in data/samples.csv. Each row contains a label and 42 numeric features. The supplied CSV is only a template; users must collect their own samples.

## 8. Privacy and Security
- The application is designed for local processing.
- Camera images are not intentionally uploaded to a remote service.
- Users should obtain consent before collecting identifiable data.
- Sample data should not contain private or sensitive information.

## 9. Evaluation Plan
Evaluate the prototype using separate test samples that were not used during training. Record:
- Overall classification accuracy
- Precision and recall for each gesture label
- Confusion matrix
- Prediction response time
- User feedback on usability and accessibility

No performance results are claimed until testing is completed.

## 10. Limitations
- Recognizes only user-trained static gestures.
- Uses one hand at a time.
- Does not recognize continuous signing or full sentences.
- Does not interpret facial expressions or sign-language grammar.
- Results depend on training data, lighting, camera quality, and hand position.
- It is not a complete or validated Indian Sign Language translator.

## 11. Future Enhancements
- Build a reviewed dataset with permission from sign-language users.
- Add support for dynamic and two-handed signs.
- Evaluate performance with qualified Indian Sign Language experts.
- Improve regional-language speech support.
- Conduct accessibility and usability testing.

## 12. Conclusion
This prototype explores how computer vision and machine learning may support inclusive communication. Further dataset development, expert evaluation, and accessibility testing are required before practical deployment.
