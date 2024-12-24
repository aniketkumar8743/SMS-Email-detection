# SMS/Email Spam Detection Using Multi-Algorithm Approach
This project implements a multi-algorithm machine learning (ML) approach for detecting spam emails and SMS messages. It leverages 10 different machine learning algorithms to classify text messages as either spam or non-spam. The top-performing algorithm is then integrated into a GUI-based system to allow users to classify emails/SMS messages effectively.

## Overview
Spam detection is a crucial task in today’s digital age due to the overwhelming volume of unwanted messages we receive daily. This project employs ten machine learning algorithms to detect spam and non-spam messages in both emails and SMS. These algorithms are evaluated based on accuracy and precision metrics, helping to identify the best performing model. The final model is then integrated into a user-friendly graphical user interface (GUI), allowing users to easily classify messages.

## Project Goals
**Experiment with multiple ML algorithms**: Test the effectiveness of ten machine learning algorithms in detecting spam messages.
**Compare performance**: Evaluate each algorithm's accuracy and precision and determine the best-performing model.
**Develop a GUI-based system**: Implement a GUI for easy interaction where users can classify SMS/emails as spam or non-spam.
**Improve spam detection**: Enhance spam filtering by identifying the most accurate and precise machine learning model for spam detection.
**Analyze strengths and weaknesses**: Study how each algorithm performs on a text-based dataset to understand the impact on spam detection.
## Dataset
The dataset used for training and evaluating the models consists of labeled text data containing both spam and non-spam (ham) messages. Each message is classified into one of the two categories, and it may contain features such as:

**Message content**: The actual text of the message.
**Label**: Indicates whether the message is spam (1) or not (0).
The dataset used in this project can be sourced from publicly available SMS spam datasets or custom data collected for spam detection.

Model Architecture
We utilize the following machine learning algorithms for spam detection:

**Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Naive Bayes
Decision Tree
Random Forest
AdaBoost Classifier
Extra Trees Classifier
Gradient Boosting Classifier
XGBoost Classifier**
Each model is trained on the text data using TF-IDF Vectorization for converting text into numerical features that can be processed by the machine learning models.

### Training Process
**Data Preprocessing:**
Text data is cleaned and preprocessed by removing stop words, punctuation, and non-alphabetic characters.
The text is converted into numerical format using TF-IDF Vectorizer.
**Model Training**:
Each model is trained using a training set of labeled data.
Hyperparameters are tuned using cross-validation to optimize the model's performance.
**Model Evaluation**
The models are evaluated using the following metrics:

**Accuracy**: The percentage of correctly classified messages.
**Precision**: The percentage of true positive results out of all predicted positives.
The performance of each model is compared, and the best-performing model based on these metrics is selected.

## Usage
### Steps to Use the System:
**Run the GUI-based Application:**

Once the system is set up, open the GUI where users can input email/SMS messages.
The system will classify the input message as "spam" or "non-spam" using the trained model.
**Input Message:**

Enter a message or email content into the provided text box.
**Classify the Message:**

Press the "Classify" button to determine if the message is spam or non-spam.
The system will output the classification result.

 ## Results
After evaluating all 10 machine learning algorithms, the following results were obtained:

**Logistic Regression**: 95% accuracy, 95% precision
**SVM**: 97.0% accuracy, 97% precision
**KNN**: 90.2% accuracy, 91% precision
**Naive Bayes**: 97.3% accuracy, 97.17% precision
**Decision Tree**: 92.7% accuracy, 92% precision
**Random Forest**: 97% accuracy, 97.0% precision
**AdaBoost**: 96.5% accuracy, 95.0% precision
**Extra Trees**: 95.2% accuracy, 95.5% precision
**Gradient Boosting**: 94.8% accuracy, 94.0% precision
**XGBoost**: 96.99.0% accuracy, 96.59% precision
The **XGBoost Classifier** emerged as the best-performing model with an accuracy of 96.990% and a precision of 96.59%, making it the model of choice for integration into the GUI-based system if we add more data it accuracy also will increased.

## Conclusion
This project provides a comprehensive analysis of spam email and SMS detection using multiple machine learning algorithms. By comparing different models, we identified the XGBoost Classifier as the best performer, which was then integrated into a user-friendly system for real-time spam classification.

The results show significant improvement in spam detection accuracy, and the developed GUI allows users to classify emails/SMS effectively, reducing the impact of unwanted messages on individuals and organizations.


