from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from psycopg2 import sql
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
import requests
import base64
from models.ECGNN import ECGNN  # Import your model here
from models.ECGClassifier import ECGClassifier 
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import matplotlib.pyplot as plt
import time
import io

app = Flask(__name__)
app.secret_key =   'your_secret_key'  # Needed for flash messages
# Database connection parameters
db_config = {
    'dbname': 'client',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost'
}

def get_db_connection():
    conn = psycopg2.connect(**db_config)
    return conn

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['signupUsername']
    email = request.form['signupEmail']
    password = request.form['signupPassword']
    
    hashed_password = generate_password_hash(password)

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Attempt to insert the new user
        cur.execute('INSERT INTO client (username, email, password) VALUES (%s, %s, %s)', 
                    (username, email, hashed_password))
        conn.commit()
        cur.close()
        conn.close()
        return 'Signup successful! Please log in.', 200  # Return success message
    except psycopg2.IntegrityError:
        conn.rollback()  # Rollback transaction on error
        return 'Error: Email already exists. Please use a different email.', 400  # Custom error message
    except Exception as e:
        print(e)
        return 'An unexpected error occurred during signup.', 500  # Return server error message

@app.route('/login', methods=['POST'])
def login():
    email = request.form['loginEmail']
    password = request.form['loginPassword']
    print(email,password)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT password FROM client WHERE email = %s', (email,))
        user = cur.fetchone()
        print(user)
        print(generate_password_hash(password))
        cur.close()
        conn.close()

        if user and check_password_hash(user[0], password):
            
            return 'Login successful!', 200  # Return success message
            
            
        else:
            return 'Invalid email or password.', 401  # Return unauthorized message
    except Exception as e:
        print(e)
        return 'An error occurred during login.', 500  # Return server error message


@app.route('/profile')
def profile():
    return render_template('profile.html')
@app.route('/get_profile', methods=['GET'])
def get_profile():
    user_id = 1  # Replace with logic to get the logged-in user's ID
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, email, phone, dob, blood_group, weight, height, bmi, allergies, conditions, medications FROM users WHERE id = %s", (user_id,))
    profile = cur.fetchone()
    cur.close()
    conn.close()
    
    if profile:
        return jsonify({
            'full_name': profile[0],
            'email': profile[1],
            'phone': profile[2],
            'dob': profile[3],
            'blood_group': profile[4],
            'weight': profile[5],
            'height': profile[6],
            'bmi': profile[7],
            'allergies': profile[8],
            'conditions': profile[9],
            'medications': profile[10]
        })
    return jsonify({'error': 'Profile not found'}), 404

@app.route('/update_profile', methods=['POST'])
def update_profile():
    data = request.get_json()  # Get the JSON data sent from the client
    # Process the data (e.g., update the database)
    # For demonstration, let's assume we just return the received data
    if data:
        return jsonify({"message": "Profile updated successfully", "data": data}), 200
    return jsonify({"message": "No data provided"}), 400

@app.route('/main')
def main():
    return render_template('main.html')

def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def load_quantized_weights_to_float_model(quantized_model_path, float_model):
    try:
        quantized_state_dict = torch.load(quantized_model_path)
        float_state_dict = {k: v for k, v in quantized_state_dict.items() if 'scale' not in k and 'zero_point' not in k and '_packed_params' not in k}
        float_model.load_state_dict(float_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading quantized weights: {e}")

# Create instances of your models
heart_rhythm_model = ECGNN()  # Replace with your actual model class
heart_rhythm_model = quantize_model(heart_rhythm_model)
load_quantized_weights_to_float_model(r'E:\Me_Personal\Evolumin\HealthMonitor\ecg_model.pth', heart_rhythm_model)
heart_rhythm_model.eval()

af_model = ECGClassifier()  # Replace with your actual model class for AF
af_model = quantize_model(af_model)
load_quantized_weights_to_float_model(r'E:\Me_Personal\Evolumin\HealthMonitor\ecg_model_arr.pth', af_model)
af_model.eval()

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    if df.isnull().values.any():
        raise ValueError("Input data contains NaN values.")
    x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def load_data_for_arrhythmia(file_path):
    df = pd.read_csv(file_path).drop(columns=['record'], errors='ignore')
    if df.isnull().values.any():
        raise ValueError("Input data contains NaN values.")
    
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    label_encoder = LabelEncoder()
    df['type'] = label_encoder.fit_transform(df['type'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=['type']))
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(df['type'].values, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=16, shuffle=True)


@app.route('/predict')
def predict():
    # Load random signal for heart rhythm
    heart_rhythm_loader = load_data_from_csv(r'E:\Me_Personal\Evolumin\HealthMonitor\ecg_local.csv')
    random_signal_hr = random.choice(list(heart_rhythm_loader.dataset.tensors[0].numpy()))


    # Predict heart rhythm
    with torch.no_grad():
        heart_rhythm_model_input = torch.tensor(random_signal_hr, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        heart_rhythm_output = heart_rhythm_model(heart_rhythm_model_input)
        heart_rhythm_prediction = 'Normal' if heart_rhythm_output.item() < 0.5 else 'Abnormal'
    print(f"Heart Rhythm Prediction: {heart_rhythm_prediction}")

    # Load random signal for AF classification
    af_loader = load_data_for_arrhythmia(r'E:\Me_Personal\Evolumin\HealthMonitor\MIT-BIH Arrhythmia Database_client.csv')
    random_signal_af = random.choice(list(af_loader.dataset.tensors[0].numpy()))

    # Predict AF group
    with torch.no_grad():
        af_model_input = torch.tensor(random_signal_af, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        af_output = af_model(af_model_input)
        af_group = torch.argmax(af_output, dim=1).item()  # Get index of maximum logit
        
        if af_group == 0:
            af_prediction = 'Group N (Non-terminating AF)'
        elif af_group == 1:
            af_prediction = 'Group S (Terminates after 1 minute)'
        else:
            af_prediction = 'Group T (Terminates immediately)'
    print(af_group)
    print(f"AF Prediction: {af_prediction}")

    # Plotting the ECG waveform
    plt.figure(figsize=(10, 4))
    plt.plot(random_signal_hr, color='blue')
    plt.title('ECG Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to avoid display
    buf.seek(0)
    
    # Encode the image in base64
    plot_image = base64.b64encode(buf.read()).decode('utf-8')
    plot_image_tag = f"data:image/png;base64,{plot_image}"

    # Render results in HTML
    return render_template('main.html', heart_rhythm=heart_rhythm_prediction, af_group=af_prediction, plot_image=plot_image_tag)

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/challenges')
def challenges():
    return render_template('challenges.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/medication')
def medication():
    return render_template('medication.html')

def register_client(client_id):
    try:
        response = requests.post("http://192.168.254.76:5002/register_client", json={'client_id': client_id})
        return response
    except Exception as e:
        print(f"Error registering client: {e}")

@app.route('/client')
def client():
    register_client("client_1")
    return render_template('client.html')

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create a directory for saving models
MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'client')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_FILES = {
    'ecg_model': 'ecg_model.pth',
    'ecg_model_ARR': 'ecg_model_arr.pth'  # Add other signal models as needed
}

def distillation_loss(student_output, teacher_output, target, model_key, temperature=2.0, alpha=0.5):
    soft_teacher_output = torch.softmax(teacher_output / temperature, dim=1)
    soft_student_output = torch.log_softmax(student_output / temperature, dim=1)
    loss_soft = torch.nn.KLDivLoss(reduction='batchmean')(soft_student_output, soft_teacher_output) * (temperature ** 2)
    
    if model_key == 'ecg_model':  # Binary classification
        loss_hard = torch.nn.BCELoss()(student_output.view(-1), target.view(-1))
    else:  # Multi-class classification
        loss_hard = torch.nn.CrossEntropyLoss()(student_output, target)
    
    return alpha * loss_hard + (1 - alpha) * loss_soft

# Function to get the teacher model from the server
def get_teacher_model(model_key):
    try:
        response = requests.get(f"http://192.168.254.76:5002/get_global_model/{model_key}")  
        if response.status_code == 200:
            model_data = response.json().get('model')
            teacher_state_dict = pickle.loads(base64.b64decode(model_data))
            if model_key == 'ecg_model':
                teacher_model = ECGNN()
            elif model_key == 'ecg_model_ARR':
                teacher_model = ECGClassifier()
            teacher_model.load_state_dict(teacher_state_dict)
            return teacher_model
        else:
            raise Exception("Failed to get the global model from the server.")
    except Exception as e:
        print(f"Error getting teacher model: {e}")
        return None

# Local training function with distillation
def train_with_distillation(student_model, teacher_model, data_loader, model_key, epochs=5, lr=0.01, temperature=2.0, alpha=0.5):
    optimizer = optim.SGD(student_model.parameters(), lr=lr)
    student_model.train()
    teacher_model.eval()
    
    for epoch in range(epochs):
        total_loss = 0
        for data, target in data_loader:
            optimizer.zero_grad()
            student_output = student_model(data)
            with torch.no_grad():
                teacher_output = teacher_model(data)
            loss = distillation_loss(student_output, teacher_output, target, model_key, temperature, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# Function to send the updated student model to the server
def send_student_model_update(student_model, model_key):
    try:
        model_state = student_model.state_dict()
        model_data = pickle.dumps(model_state)
        model_data_base64 = base64.b64encode(model_data).decode('utf-8')
        response = requests.post(f"http://192.168.254.76:5002/update_global_model/{model_key}", json={'model': model_data_base64})
        return response
    except Exception as e:
        print(f"Error sending model update: {e}")   

# Function to load data from CSV
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError("Input data contains NaN values.")
    x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def load_data_for_arrhythmia(file_path):
    df = pd.read_csv(file_path).drop(columns=['record'], errors='ignore')
    if df.isnull().values.any():
        raise ValueError("Input data contains NaN values.")
    
    label_encoder = LabelEncoder()
    df['type'] = label_encoder.fit_transform(df['type'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=['type']))
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(df['type'].values, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=16, shuffle=True)


# Function to quantize the model
def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def load_quantized_weights_to_float_model(quantized_model_path, float_model):
    try:
        quantized_state_dict = torch.load(quantized_model_path)
        float_state_dict = {k: v for k, v in quantized_state_dict.items() if 'scale' not in k and 'zero_point' not in k and '_packed_params' not in k}
        float_model.load_state_dict(float_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading quantized weights: {e}")

# Function to start the training process
@app.route('/start_training', methods=['POST'])
def start_training():
    model_key = request.form.get('model_type')
    if model_key not in MODEL_FILES:
        flash("Invalid model key provided.", "error")
        return redirect(url_for('client'))

    file = request.files['file_upload']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    student_model_file_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILES[model_key])

    if model_key == 'ecg_model':
        student_model = ECGNN()
        data_loader = load_data_from_csv(file_path)
    elif model_key == 'ecg_model_ARR':
        student_model = ECGClassifier()
        data_loader = load_data_for_arrhythmia(file_path)
    
    if os.path.exists(student_model_file_path):
        quantized_model = quantize_model(student_model)
        load_quantized_weights_to_float_model(student_model_file_path, quantized_model)
        print(f"{model_key} quantized model weights loaded successfully.")
    
    teacher_model = get_teacher_model(model_key)
    if teacher_model is None:
        flash("Failed to load teacher model.", "error")
        return redirect(url_for('client'))

    train_with_distillation(student_model, teacher_model, data_loader, model_key)

    quantized_student_model = quantize_model(student_model)
    torch.save(quantized_student_model.state_dict(), student_model_file_path)

    response = send_student_model_update(quantized_student_model, model_key)
    flash("Training completed and model updated successfully." if response and response.status_code == 200 else "Failed to update model on server.", "success" if response and response.status_code == 200 else "error")
    return redirect(url_for('client'))


if __name__ == '__main__':
    app.run(debug=True)
