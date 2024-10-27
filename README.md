## Medcare_FederatedLearning
Using Bio-Signals which can be acquired from a smartwatch, such as the ECG, PPG, and accelerometer, a federated model is trained to provide personalized and efficient patient care in a secure environment.

## Project Name
MedChain: Federated Learning in Personalised Patient Care

## Technologies Used
- PyTorch
- Flask
- HTML
- CSS
- JS
- Socket Programming
- Postgres

## Project Description
In this project, we are using two datasets:
1. ECG dataset kaggle, which helps classify the detected signals into 2 classes- Normal and Abnormal
2. Arrhythmia dataset kaggle, which has three classes-
               Group N: non-terminating AF (defined as AF that was not observed to have terminated for the duration of the long-term recording, at least an hour following the segment).
               Group S: AF that terminates one minute after the end of the record.
               Group T: AF that terminates immediately (within one second) after the end of the record. Note that these records come from the same long-term ECG recordings as those in Group S and immediately follow the Group S records.

We use these predictions further to train our federated model's server, using methods like distillation and quantization to reduce the model size and help update the client model's weights. 
This is then used to predict the outputs of the inputted time signal, which is selected at random here. The output is reflected in the main webpage, which can be used on our phones and any remote server.
Currently, this runs on a local server but can be further implemented on a global server (like Azure).

## Demo Video
- Link: https://youtu.be/mKTlS2_YEJw

## Installation and Setup
Step 1: Run the app.py file of both the client and server     
Step 2: Open the file on the local host on the SAME network     
Step 3: On the server side, train the model and start the server. Keep the server running to connect to the client     
Step 4: On the client side, choose the dataset to be trained and start the training.       
Step 5: On the main page of the web interface, we can now predict the ECG outputs on whether there is any abnormal heart rate or chances of arrhythmia and get the plot of the ECG signal taken in real-time.


## Future Scope
- Integration with Wearable Devices: Expand the system to incorporate data from wearable health monitoring devices for a more comprehensive view of patient health.
- Broader Healthcare Applications: Explore the application of federated learning in other healthcare domains, such as diabetes management and mental health monitoring.
- Continuous Model Improvement: Enable the model to continuously learn from new data across various settings, further enhancing its accuracy and effectiveness in real-world scenarios.
-  Lifestyle prediction: On analyzing the output provided by analyzing the signals, we can provide customized lifestyles to the user
-  This can be extended for PPG and Accelerometer data; which can give further insights on the respiratory rates, posture alerts, sleep monitoring, BP monitoring, and track physical motions, which is particularly helpful in elderly patients to help detect daily changes that may detect a decline in health or cognitive functions




