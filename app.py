from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import sys
import io
import time
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from ast import literal_eval
import tempfile
import warnings
from backend import get_results
from utils.file_utils import initialize_session_state  # Ensure this exists

# Set UTF-8 Encoding
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# Flask App Initialization
app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Initialize session state variables
initialize_session_state()

# Load Excel Data
def load_data(uploaded_file):
    """Load uploaded Excel file into a Pandas DataFrame."""
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

# Convert Data to Excel Bytes
def to_excel(dataframe):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Recommended Resources')
    output.seek(0)
    return output

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({"error": "Both files must be uploaded."}), 400

        rr_file = request.files['file1']
        bench_file = request.files['file2']
        isCvSkills = request.form.get('isCvSkills') == 'true'

        # Save temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as rr_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as bench_temp:
            
            rr_file.save(rr_temp.name)
            bench_file.save(bench_temp.name)

            rr_df = load_data(rr_temp.name)
            bench_data = load_data(bench_temp.name)

        # Process Data
        get_results(rr_df, bench_data, isCvSkills)

        # Simulate processing time
        time.sleep(5)

        return jsonify({"message": "Processing complete"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommendations/rr', methods=['GET'])
def get_recommendations_by_rr():
    try:
        bench_file_path = request.args.get('bench_file')
        rr_file_path = request.args.get('rr_file')

        if not bench_file_path or not rr_file_path:
            return jsonify({"error": "Missing required file paths"}), 400

        bench_data = load_data(bench_file_path)
        rr_file = load_data(rr_file_path)

        refined_rr_df = pd.read_excel("assets/output/refined_RR_To_Profiles_Recommendations.xlsx")

        # Process Data
        refined_rr_df.drop(["uuid"], axis=1, inplace=True)
        for col in ["RR Skills", "Candidate_Skills", "matched_skillset", "recommended_trainings"]:
            refined_rr_df[col] = refined_rr_df[col].apply(lambda x: ", ".join(literal_eval(str(x))) if pd.notna(x) else "")

        refined_rr_df["Employee Name"] = refined_rr_df["portal_id"].astype(str).apply(lambda pid: get_name(pid, bench_data))
        refined_rr_df["Match Score"] = round(refined_rr_df["Score"] * 100)
        
        # Rename Columns
        refined_rr_df.rename(columns={
            'Candidate_Skills': 'Overall Employee Skills',
            'matched_skillset': 'Matched Skills',
            'portal_id': 'Portal ID',
            'recommended_trainings': 'Recommended Trainings',
            'bench_period': 'Bench Period'
        }, inplace=True)

        rr_cols = ['RR', 'RR Skills', 'Portal ID', 'Employee Name', 'Overall Employee Skills', 
                   'Matched Skills', 'Recommended Trainings', 'Match Score', 'Bench Period']
        
        return jsonify(refined_rr_df[rr_cols].to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommendations/profiles', methods=['GET'])
def get_recommendations_by_profiles():
    try:
        bench_file_path = request.args.get('bench_file')
        rr_file_path = request.args.get('rr_file')

        if not bench_file_path or not rr_file_path:
            return jsonify({"error": "Missing required file paths"}), 400

        bench_data = load_data(bench_file_path)
        rr_file = load_data(rr_file_path)

        refined_profile_df = pd.read_excel("assets/output/refined_Profiles_To_RR_Recommendations.xlsx")

        # Process Data
        refined_profile_df.drop(["uuid"], axis=1, inplace=True)
        for col in ["RR Skills", "Candidate Skills", "matched_skillset", "recommended_trainings"]:
            refined_profile_df[col] = refined_profile_df[col].apply(lambda x: ", ".join(literal_eval(str(x))) if pd.notna(x) else "")

        refined_profile_df["Employee Name"] = refined_profile_df["portal_id"].astype(str).apply(lambda pid: get_name(pid, bench_data))
        refined_profile_df["Match Score"] = round(refined_profile_df["Score"] * 100)

        # Rename Columns
        refined_profile_df.rename(columns={
            'Candidate Skills': 'Overall Employee Skills',
            'matched_skillset': 'Matched Skills',
            'portal_id': 'Portal ID',
            'recommended_trainings': 'Recommended Trainings'
        }, inplace=True)

        profile_cols = ['Portal ID', 'Employee Name', 'Overall Employee Skills', 'RR', 'RR Skills', 
                        'Matched Skills', 'Recommended Trainings', 'Match Score']

        return jsonify(refined_profile_df[profile_cols].to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_name(pid, bench_data):
    """Retrieve employee name by Portal ID."""
    bench_data["PID"] = bench_data["PID"].astype(str)
    res = bench_data.loc[bench_data["PID"] == pid, "EE Name"]
    return res.values[0] if not res.empty else "Not Available"

if __name__ == '__main__':
    app.run(debug=True)
