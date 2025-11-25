import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def analyze_file(file_path):
    """
    Analyze eye tracker data from the given file path.
    
    Args:
        file_path (str): Path to the input file (Excel or CSV)
        
    Returns:
        str: Path to the generated PDF report
    """
    # Define AOIs (Areas of Interest)
    AOIs = {
        'Top-Left': ((0, 0), (960, 540)),
        'Bottom-Right': ((960, 540), (1920, 1080)),
    }

    def get_aoi(x, y):
        """Determine which AOI a point belongs to."""
        for label, ((x1, y1), (x2, y2)) in AOIs.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return label
        return "Outside"

    def load_data(file_path):
        """Load data from Excel or CSV file."""
        try:
            return pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            print(f"Excel read failed, trying CSV. Reason: {e}")
            return pd.read_csv(file_path)

    # Create results directory
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    results_dir = os.path.join(file_dir, f"{os.path.splitext(file_name)[0]}_results")
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Load and preprocess data
        df = load_data(file_path)
        df.columns = df.columns.str.strip().str.lower()
        print("Normalized columns:", df.columns.tolist())

        # Ensure required columns are present and numeric
        required_cols = ['x', 'y', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        # Convert to numeric and drop rows with invalid data
        for col in ['x', 'y', 'timestamp']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['x', 'y', 'timestamp'], inplace=True)

        if df.empty:
            raise ValueError("No valid numeric data found in the file")

        # Fixation Analysis
        df['fixation'] = ((df['x'].diff().abs() < 30) & (df['y'].diff().abs() < 30)).astype(int)
        df['fixation_group'] = (df['fixation'] != df['fixation'].shift()).cumsum()
        fixation_durations = df[df['fixation'] == 1].groupby('fixation_group').size()
        fixation_avg = fixation_durations.mean()

        # Saccade Analysis
        df['saccade'] = (df['fixation'] == 0).astype(int)
        df['saccade_amplitude'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2)

        # AOI Analysis
        df['AOI'] = df.apply(lambda row: get_aoi(row['x'], row['y']), axis=1)
        aoi_counts = df['AOI'].value_counts().to_dict()

        # Time-to-Event Analysis
        if 'event' in df.columns:
            event_time = df[df['event'] == 1]['timestamp'].min()
            post_event = df[df['timestamp'] >= event_time]
            first_aoi_hit = post_event[post_event['AOI'] != "Outside"]['timestamp'].min()
            time_to_event = first_aoi_hit - event_time if pd.notnull(first_aoi_hit) else "N/A"
        else:
            time_to_event = "No event column"

        # Generate Visualizations
        # Saccade Amplitude Plot
        plt.figure()
        plt.plot(df['saccade_amplitude'])
        plt.title("Saccade Amplitudes")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        plt.savefig(os.path.join(results_dir, "saccade_plot.png"))
        plt.close()

        # Heatmap
        heatmap, _, _ = np.histogram2d(df['x'], df['y'], bins=(64, 36))
        plt.imshow(heatmap.T, origin='lower', cmap='hot')
        plt.title("Gaze Heatmap")
        plt.savefig(os.path.join(results_dir, "heatmap.png"))
        plt.close()

        # Gaze Plot
        plt.scatter(df['x'], df['y'], alpha=0.5)
        plt.plot(df['x'], df['y'], color='gray', alpha=0.3)
        plt.title("Gaze Plot")
        plt.savefig(os.path.join(results_dir, "gaze_plot.png"))
        plt.close()

        # Scan Path
        plt.figure(figsize=(10, 6))
        for i in range(len(df)-1):
            plt.plot([df['x'].iloc[i], df['x'].iloc[i+1]], 
                    [df['y'].iloc[i], df['y'].iloc[i+1]], color='blue')
        plt.title("Scan Path")
        plt.savefig(os.path.join(results_dir, "scan_path.png"))
        plt.close()

        # Machine Learning Analysis
        df_ml = df[df['AOI'] != "Outside"]
        if not df_ml.empty:
            X = df_ml[['x', 'y']]
            y = df_ml['AOI']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            ml_report = classification_report(y_test, y_pred)
        else:
            ml_report = "Not enough AOI data for ML."

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add report content
        pdf.cell(0, 10, "EyeTracker Analysis Report", ln=True)
        pdf.cell(0, 10, f"File: {file_name}", ln=True)

        pdf.cell(0, 10, "--- Fixation Analysis ---", ln=True)
        pdf.cell(0, 10, f"Average Fixation Duration: {fixation_avg:.2f} frames", ln=True)

        pdf.cell(0, 10, "--- AOI Analysis ---", ln=True)
        for k, v in aoi_counts.items():
            pdf.cell(0, 10, f"{k}: {v} points", ln=True)

        pdf.cell(0, 10, "--- Time to Event ---", ln=True)
        pdf.cell(0, 10, f"{time_to_event}", ln=True)

        pdf.cell(0, 10, "--- ML Report ---", ln=True)
        for line in ml_report.split('\n'):
            if line.strip():
                pdf.multi_cell(0, 10, line.strip())

        # Add plots to PDF
        for img in ["saccade_plot.png", "heatmap.png", "gaze_plot.png", "scan_path.png"]:
            pdf.add_page()
            pdf.image(os.path.join(results_dir, img), x=10, y=20, w=180)

        # Save PDF
        pdf_path = os.path.join(results_dir, "report.pdf")
        pdf.output(pdf_path)
        print("✅ Results saved in:", results_dir)

        return pdf_path

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise
