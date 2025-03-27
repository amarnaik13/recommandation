from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import sys
import io
import time
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from utils.file_utils import *
from ast import literal_eval
import tempfile
from backend import get_results

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

app= Flask(__name__, static_folder='build', static_url_path='/')
#app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
  return send_from_directory(app.static_folder, 'index.html')

# Initialize session state variables
initialize_session_state()

# load the excel data
def load_data(uploaded_file):
    """Load uploaded Excel file into a Pandas DataFrame."""
    return pd.read_excel(uploaded_file)

# download to excel
def to_excel(dataframe):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Recommended Resources')
        writer.close()
    processed_data = output.getvalue()
    return processed_data

# Color the column
def highlighter(val):
    color = '#ADFF2F'
    return f'background-color: {color}'

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        rr_file = request.files['file1']
        bench_file = request.files['file2']
        isCvSkills = request.form.get('isCvSkills') == 'true'
        
        
        rr_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        bench_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        print(f"isCvSkills: {isCvSkills}")
        rr_file.save(rr_temp.name)
        bench_file.save(bench_temp.name)
        
        rr_df = load_data(rr_temp.name)
        bench_data = load_data(bench_temp.name)

        # Making API calls for generation        
        get_results(rr_df, bench_data,isCvSkills)

        # Simulate processing time
        time.sleep(5)

        # Return success response with temporary file paths
        return jsonify({"file1": rr_temp.name, "file2": bench_temp.name}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommendations/rr', methods=['GET'])
def get_recommendations_by_rr():
    try:
        bench_data = load_data(request.args.get('bench_file'))
        rr_file = load_data(request.args.get('rr_file'))

        refined_rr_df = pd.read_excel(r"assets/output/refined_RR_To_Profiles_Recommendations.xlsx")
        refined_rr_df.drop(["uuid"], axis=1, inplace=True)

        def remove_list(x):
            return ", ".join(literal_eval(x))

        refined_rr_df["RR Skills"] = refined_rr_df["RR Skills"].apply(remove_list)
        refined_rr_df["Candidate_Skills"] = refined_rr_df["Candidate_Skills"].apply(remove_list)
        refined_rr_df["matched_skillset"] = refined_rr_df["matched_skillset"].apply(remove_list)
        refined_rr_df["recommended_trainings"] = refined_rr_df["recommended_trainings"].apply(remove_list)

        refined_rr_df["RR"] = refined_rr_df["RR"].astype(str)
        refined_rr_df["portal_id"] = refined_rr_df["portal_id"].astype(str)
        refined_rr_df["Employee Name"] = refined_rr_df["portal_id"].apply(lambda pid: get_name(pid, bench_data))
        refined_rr_df["bench_period"] = refined_rr_df["bench_period"].astype(str)
        refined_rr_df["Score"] = round(refined_rr_df["Score"] * 100)

        refined_rr_df.rename(columns={
            'Candidate_Skills': 'Overall Employee Skills',
            'Score': 'Match Score',
            'bench_period': 'Bench Period',
            'matched_skillset': 'Matched Skills',
            'portal_id': 'Portal ID',
            'recommended_trainings': 'Recommended Trainings',
        }, inplace=True)

        rr_cols = ['RR', 'RR Skills', 'Portal ID', 'Employee Name', 'Overall Employee Skills', 'Matched Skills', 'Recommended Trainings', 'Match Score', 'Bench Period','Profile Link']
        styled_rr_df = refined_rr_df[rr_cols]

        return jsonify(styled_rr_df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommendations/profiles', methods=['GET'])
def get_recommendations_by_profiles():
    try:
        bench_data = load_data(request.args.get('bench_file'))
        rr_file = load_data(request.args.get('rr_file'))

        refined_profile_df = pd.read_excel(r"assets/output/refined_Profiles_To_RR_Recommendations.xlsx")
        refined_profile_df.drop(["uuid"], axis=1, inplace=True)

        def remove_list(x):
            return ", ".join(literal_eval(x))

        refined_profile_df["RR Skills"] = refined_profile_df["RR Skills"].apply(remove_list)
        refined_profile_df["Candidate Skills"] = refined_profile_df["Candidate Skills"].apply(remove_list)
        refined_profile_df["matched_skillset"] = refined_profile_df["matched_skillset"].apply(remove_list)
        refined_profile_df["recommended_trainings"] = refined_profile_df["recommended_trainings"].apply(remove_list)

        refined_profile_df["RR"] = refined_profile_df["RR"].astype(str)
        refined_profile_df["portal_id"] = refined_profile_df["portal_id"].astype(str)
        refined_profile_df["Employee Name"] = refined_profile_df["portal_id"].apply(lambda pid: get_name(pid, bench_data))
        refined_profile_df["Score"] = round(refined_profile_df["Score"] * 100)

        refined_profile_df.rename(columns={
            'Candidate Skills': 'Overall Employee Skills',
            'Score': 'Match Score',
            'matched_skillset': 'Matched Skills',
            'portal_id': 'Portal ID',
            'recommended_trainings': 'Recommended Trainings',
        }, inplace=True)

        profile_cols = ['Portal ID', 'Employee Name', 'Overall Employee Skills', 'RR', 'RR Skills', 'Matched Skills', 'Recommended Trainings', 'Match Score','Profile Link']
        styled_profile_df = refined_profile_df[profile_cols]

        return jsonify(styled_profile_df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_name(pid, bench_data):
    bench_data["PID"] = bench_data["PID"].astype(str)
    res = bench_data[bench_data["PID"] == pid]["EE Name"]
    if not res.empty:
        res = res.values[0]
    else:
        res = "Not Available"
    return res

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8000)