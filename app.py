from flask import Flask, request, jsonify, send_file, render_template
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
from analytics import analyze_file
import csv
import pandas as pd

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['USERS_CSV'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.csv')  # Absolute path to users.csv

# Configuration
ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'csv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_user_to_csv(email, password, name):
    """Save user data to CSV file"""
    try:
        file_exists = os.path.isfile(app.config['USERS_CSV'])
        print(f"CSV file exists: {file_exists}")
        print(f"Attempting to save user data to: {os.path.abspath(app.config['USERS_CSV'])}")
        
        with open(app.config['USERS_CSV'], 'a', newline='') as csvfile:
            fieldnames = ['Email', 'Password', 'Name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                print("Creating new CSV file with headers")
                writer.writeheader()
            
            writer.writerow({
                'Email': email,
                'Password': password,
                'Name': name
            })
            print(f"Successfully saved user data for: {email}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")
        raise e

def check_user_exists(email):
    """Check if user already exists in CSV"""
    if not os.path.isfile(app.config['USERS_CSV']):
        return False
    
    try:
        df = pd.read_csv(app.config['USERS_CSV'])
        return email in df['Email'].values
    except:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        print("Received registration request")
        data = request.get_json()
        print(f"Received data: {data}")
        
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        
        if not all([email, password, name]):
            print("Missing required fields")
            return jsonify({'error': 'Missing required fields'}), 400
        
        if check_user_exists(email):
            print(f"Email already exists: {email}")
            return jsonify({'error': 'Email already registered'}), 400
        
        try:
            save_user_to_csv(email, password, name)
            print(f"Successfully registered user: {email}")
            return jsonify({'message': 'Registration successful'}), 200
        except Exception as e:
            print(f"Error saving user data: {str(e)}")
            return jsonify({'error': f'Failed to save user data: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Unexpected error in registration: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        print("Received login request")
        data = request.get_json()
        print(f"Login attempt for email: {data.get('email')}")
        
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            print("Missing email or password")
            return jsonify({'error': 'Missing email or password'}), 400
        
        try:
            if not os.path.isfile(app.config['USERS_CSV']):
                print("Users CSV file does not exist")
                return jsonify({'error': 'Invalid credentials'}), 401
            
            df = pd.read_csv(app.config['USERS_CSV'])
            print(f"Found {len(df)} users in CSV")
            print(f"Checking credentials for email: {email}")
            
            user = df[(df['Email'] == email) & (df['Password'] == password)]
            print(f"Found matching user: {len(user) > 0}")
            
            if len(user) > 0:
                print(f"Login successful for: {email}")
                return jsonify({'message': 'Login successful', 'email': email}), 200
            else:
                print("Invalid credentials")
                return jsonify({'error': 'Invalid credentials'}), 401
        except Exception as e:
            print(f"Error during login: {str(e)}")
            return jsonify({'error': 'Login failed'}), 500
            
    except Exception as e:
        print(f"Unexpected error in login: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Create results directory
        results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Run analysis
        try:
            pdf_path = analyze_file(file_path)
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

        # Get the base directory of the results
        results_base = os.path.dirname(pdf_path)
        
        # Prepare response data
        response_data = {
            'heatmap_url': f'/results/{os.path.basename(results_base)}/heatmap.png',
            'gaze_plot_url': f'/results/{os.path.basename(results_base)}/gaze_plot.png',
            'scan_path_url': f'/results/{os.path.basename(results_base)}/scan_path.png',
            'saccade_plot_url': f'/results/{os.path.basename(results_base)}/saccade_plot.png',
            'pdf_url': f'/results/{os.path.basename(results_base)}/report.pdf',
            'metrics': get_metrics_from_pdf(pdf_path)
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/results/<path:filename>')
def serve_result(filename):
    results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    file_path = os.path.join(results_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path)

def get_metrics_from_pdf(pdf_path):
    # This is a placeholder function. In a real implementation,
    # you would extract metrics from the PDF or store them separately
    return """Fixation Analysis:
- Average Fixation Duration: 120ms
- Total Fixations: 45

AOI Analysis:
- Top-Left: 25 points
- Bottom-Right: 30 points
- Outside: 10 points

Saccade Analysis:
- Average Amplitude: 150px
- Total Saccades: 40

Time to Event: 2.5s"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)