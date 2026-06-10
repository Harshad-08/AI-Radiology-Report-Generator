import os
import json
import pandas as pd
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

def ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)

def save_report(findings, impression, recommendations, image_name, pdf_path):
    """
    Saves a generated radiology report as a JSON file in reports/ directory.
    Returns:
        str: Path to the saved JSON report.
    """
    ensure_reports_dir()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"CXR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    report_data = {
        "id": report_id,
        "timestamp": timestamp,
        "image_name": os.path.basename(image_name),
        "findings": findings,
        "impression": impression,
        "recommendations": recommendations,
        "pdf_path": pdf_path
    }
    
    json_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
        
    return json_path

def get_report_history():
    """
    Retrieves all stored JSON reports, sorted by timestamp descending.
    Returns:
        list of dict: List of reports.
    """
    ensure_reports_dir()
    
    reports = []
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith(".json"):
            json_path = os.path.join(REPORTS_DIR, filename)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    reports.append(data)
            except Exception as e:
                print(f"Error reading JSON report {filename}: {str(e)}")
                
    # Sort descending by timestamp
    reports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return reports

def get_report_history_dataframe():
    """
    Returns the report history as a Pandas DataFrame for dashboard rendering.
    """
    history = get_report_history()
    if not history:
        return pd.DataFrame(columns=["Report ID", "Timestamp", "Image Name", "Impression"])
        
    df_data = []
    for item in history:
        df_data.append({
            "Report ID": item.get("id", ""),
            "Timestamp": item.get("timestamp", ""),
            "Image Name": item.get("image_name", ""),
            "Impression": item.get("impression", "").replace("\n", " ")
        })
        
    return pd.DataFrame(df_data)

def load_report_by_id(report_id):
    """
    Loads a specific report data dictionary by report ID.
    """
    ensure_reports_dir()
    json_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
