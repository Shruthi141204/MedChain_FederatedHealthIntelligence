from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
import os
import base64
from models.ECGNN import ECGNN  # Import the ECG model
from models.ECGClassifier import ECGClassifier 
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key for session management

# PostgreSQL Database Configuration
db_config = {
    'dbname': 'evolumin',
    'user': 'postgres',
    'password': 'Database',
    'host': 'localhost'
}

# Define a dictionary that maps model keys to their respective model classes
MODEL_FILES = {
    'ecg_model': 'ecg_model.pth',
    'ecg_model_ARR': 'ecg_model_arr.pth'  # Add other signal models as needed
}

# Define a dictionary mapping each model key to its corresponding model class
MODEL_CLASSES = {
    'ecg_model': ECGNN,          # Model for basic ECG signal processing
    'ecg_model_ARR': ECGClassifier  # Model for arrhythmia classification
}

# Initialize the global models dictionary to hold each model instance
global_models = {
    key: MODEL_CLASSES[key]() for key in MODEL_FILES.keys()
}

client_updates = {key: [] for key in MODEL_FILES.keys()}  # Store updates for each model
AGGREGATION_THRESHOLD = 1  # Number of client updates needed before aggregation
listening = False  # Initialize listening state

# Load existing models if they exist
for key, model in global_models.items():
    if os.path.exists(MODEL_FILES[key]):
        model.load_state_dict(torch.load(MODEL_FILES[key]))
        print(f"Loaded existing model for {key}.")
    else:
        print(f"Created a new model for {key}.")

# Load data from CSV file
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Labels
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ensure labels have the correct shape
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=16, shuffle=True)

# Load data from CSV file for `ecg_model_ARR` (arrhythmia classification model)
def load_data_for_arrhythmia(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['record'])  # Drop any unwanted columns, e.g., 'record'

    # Encode target labels and standardize features
    label_encoder = LabelEncoder()
    df['type'] = label_encoder.fit_transform(df['type'])

    X = df.drop(columns=['type']).values  # Features
    y = df['type'].values  # Labels

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)  # Use long dtype for classification labels

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=16, shuffle=True)

# Train the global model using server-side dataset
def train_global_model(server_loader, model_key):
    global_model = global_models[model_key]
    print(model_key)

    if model_key=="ecg_model_ARR":
        criterion=torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    else:
        criterion = torch.nn.BCELoss()  # Binary Cross Entropy for binary classification
        optimizer = optim.SGD(global_model.parameters(), lr=0.01)

    global_model.train()
    for epoch in range(5):  # Train for 5 epochs
        total_loss = 0
        for batch_idx, (data, target) in enumerate(server_loader):
            optimizer.zero_grad()
            output = global_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Global Model Training - {model_key} - Epoch {epoch+1}, Loss: {total_loss / len(server_loader)}")

    print(f"{model_key.capitalize()} model training completed.")

# Aggregate client updates into the global model
def aggregate(model_key):
    global_model = global_models[model_key]
    updates = client_updates[model_key]
    
    if not updates:
        print("No client updates to aggregate.")
        return

    global_dict = global_model.state_dict()
    
    # Initialize an empty dictionary to accumulate the averaged updates
    avg_dict = {k: torch.zeros_like(v) for k, v in global_dict.items()}

    # Sum up the weights from all clients
    for client_state in updates:
        for k in avg_dict.keys():
            if k in client_state:
                avg_dict[k] += client_state[k]

    # Take the average of the updates
    for k in avg_dict.keys():
        avg_dict[k] /= len(updates)

    # Load the averaged weights into the global model
    global_model.load_state_dict(avg_dict)

    print(f"Aggregation completed and global model for {model_key} updated.")

# Online clients list
online_clients = []

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        admin_email = request.form['admin_email']
        password = request.form['password']

        # Database connection
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            query = "SELECT * FROM admin WHERE admin_email = %s"
            cursor.execute(query, (admin_email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            # Check if user exists and if the password matches
            if user and check_password_hash(user[3], password):  # Assuming password is the 4th column
                session['admin_email'] = admin_email
                return redirect(url_for('dashboard'))
            else:
                return "Invalid email or password", 401
        except Exception as e:
            return f"Database error: {str(e)}", 500

    return render_template('login.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'admin_email' not in session:
        return redirect(url_for('login'))
    
    model_status = {key: "Updating" if len(client_updates[key]) >= AGGREGATION_THRESHOLD else "Idle" for key in MODEL_FILES.keys()}
    return render_template('dashboard.html', model_status=model_status, online_clients=online_clients, listening=listening)

# Toggle listening state
@app.route('/toggle_listening', methods=['POST'])
def toggle_listening():
    global listening
    listening = not listening  # Toggle the listening state
    return jsonify({"listening": listening}), 200

# Logout route
@app.route('/logout')
def logout():
    session.pop('admin_email', None)
    return redirect(url_for('login'))

# Endpoint to receive the global model request from the client
@app.route('/get_global_model/<model_key>', methods=['GET'])
def send_global_model(model_key):
    try:
        if model_key not in MODEL_FILES:
            return jsonify({'error': "Invalid model key provided."}), 400

        if not listening:
            return jsonify({'error': "Server is not currently listening for updates."}), 503
        
        # Log the request for debugging purposes
        print(f"Sending global model for: {model_key}")

        # Serialize the model state dictionary
        model_data = pickle.dumps(global_models[model_key].state_dict())
        
        # Encode model data to Base64 to safely transfer via JSON
        model_data_base64 = base64.b64encode(model_data).decode('utf-8')

        return jsonify({'model': model_data_base64}), 200

    except Exception as e:
        # Log the error for debugging
        print(f"Error sending global model: {str(e)}")
        return jsonify({'error': f"Failed to send global model: {str(e)}"}), 500


# Endpoint for clients to send their model updates
@app.route('/update_global_model/<model_key>', methods=['POST'])  # You can use PUT if you prefer
def receive_model_update(model_key):
    try:
        # Check if the server is currently listening
        if model_key not in MODEL_FILES:
            return jsonify({'error': "Invalid model key provided."}), 400

        if not listening:
            return jsonify({'error': "Server is not currently listening for updates."}), 503
        
        client_model_data = request.json['model']  # Get JSON data
        client_model_state = pickle.loads(base64.b64decode(client_model_data))  # Decode Base64 and deserialize

        # Add the client model state to the list of updates
        client_updates[model_key].append(client_model_state)

        # Perform aggregation if enough clients have sent updates
        if len(client_updates[model_key]) >= AGGREGATION_THRESHOLD:
            aggregate(model_key)
            client_updates[model_key].clear()  # Clear the list after aggregation

            # Save the updated global model
            torch.save(global_models[model_key].state_dict(), MODEL_FILES[model_key])
            print(f"Updated global model for {model_key} saved after aggregation.")

        return jsonify({"message": "Model update received successfully."}), 200
    except Exception as e:
        return jsonify({'error': f"Failed to process the model update: {e}"}), 500

# Modify the `/upload_data_and_train` route to handle model-specific data loading
@app.route('/upload_data_and_train/<model_key>', methods=['POST'])
def upload_data_and_train(model_key):
    try:
        if model_key not in MODEL_FILES:
            return jsonify({'error': "Invalid model key provided."}), 400

        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Choose the data loader based on model type
        if model_key == 'ecg_model':
            server_loader = load_data_from_csv(file_path)
        elif model_key == 'ecg_model_ARR':
            server_loader = load_data_for_arrhythmia(file_path)
        else:
            return jsonify({"error": f"No data loader available for model key: {model_key}"}), 400

        train_global_model(server_loader, model_key)

        # Save the trained global model
        torch.save(global_models[model_key].state_dict(), MODEL_FILES[model_key])
        print(f"Global model for {model_key} trained and saved.")

        return jsonify({"message": f"Global model for {model_key} trained successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load data or train the model: {e}"}), 500
    
# Endpoint for clients to register and mark themselves as online
@app.route('/register_client', methods=['POST'])
def register_client():
    client_id = request.json.get('client_id')
    if client_id and client_id not in online_clients:
        online_clients.append(client_id)
        return jsonify({"message": "Client registered successfully."}), 200
    else:
        return jsonify({"error": "Client ID is required or already registered."}), 400

# Server-side endpoint to return teacher's logits
@app.route('/get_teacher_logits', methods=['POST'])
def send_teacher_logits():
    try:
        # Decode the data sent by the client
        data_base64 = request.json.get('data')
        data = pickle.loads(base64.b64decode(data_base64))

        # Forward pass through the teacher model to get logits
        global_model = global_models['ecg']  # Assume the ECG model is used for logits
        global_model.eval()
        with torch.no_grad():
            teacher_logits = global_model(data)

        # Encode the logits to Base64 to send back to the client
        logits_encoded = base64.b64encode(pickle.dumps(teacher_logits)).decode('utf-8')
        return jsonify({'logits': logits_encoded}), 200
    except Exception as e:
        return jsonify({'error': f"Failed to send teacher logits: {e}"}), 500

# Endpoint to fetch the list of online clients
@app.route('/get_online_clients', methods=['GET'])
def get_online_clients():
    return jsonify({"clients": online_clients}), 200

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
