# AI-Powered Sign Language to Speech Translator

## 1. Project Overview

This project proposes an AI-powered application that recognizes a limited set of hand gestures and converts them into text and speech. It aims to support more inclusive communication for deaf and hard-of-hearing individuals.

## 2. Problem Statement

Deaf and hard-of-hearing individuals may face communication barriers in hospitals, educational institutions, government offices, and public places. This project explores how computer vision and machine learning can help reduce these barriers.

## 3. Objectives

- Recognize selected hand gestures using a camera.
- Display predicted gesture labels as text.
- Convert predicted labels into speech.
- Provide a simple and affordable prototype.
- Explore accessible communication through AI.

## 4. Proposed Technology Stack

- Python
- Streamlit for the user interface
- OpenCV for image processing
- MediaPipe for hand landmark detection
- Scikit-learn for gesture classification
- pyttsx3 for text-to-speech

## 5. How It Works

1. The user selects a label for a gesture.
2. The camera captures the hand gesture.
3. MediaPipe extracts hand landmarks.
4. The application stores labeled landmark samples.
5. A machine-learning model learns from the collected samples.
6. The model predicts a label for a new gesture.
7. The predicted label is displayed and can be spoken aloud.

## 6. Current Scope and Limitations

This is a proposed starter prototype for a small set of user-trained static gestures. It is not a complete or validated Indian Sign Language translator. It does not yet interpret continuous signing, facial expressions, or sign-language grammar. Recognition accuracy must be evaluated using separate test data.

## 7. Privacy

The application is designed to process camera images locally. Sample landmark data is stored locally unless the user chooses to share it. Do not upload personal or identifiable data without consent.

## 8. Installation

Python 3.10 or 3.11 is recommended.

Create and activate a virtual environment:

    python -m venv .venv

On Windows:

    .venv\Scripts\activate

On macOS or Linux:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

## 9. Testing

Run the included tests with:

    pytest -q

## 10. Future Improvements

- Support a broader, expert-reviewed Indian Sign Language vocabulary.
- Evaluate recognition accuracy and response time.
- Explore two-hand and motion-based gestures.
- Improve regional language and accessibility support.
- Conduct usability testing with Deaf and hard-of-hearing users and qualified sign-language experts.

## 11. Open-Source License

This project is intended to use the MIT License. Refer to the LICENSE file for terms.

## 12. Disclaimer

This project is an educational prototype. It should not replace qualified interpreters or be used as the sole communication method for medical, legal, emergency, or other critical situations.
