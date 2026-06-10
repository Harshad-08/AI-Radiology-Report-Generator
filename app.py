import os
import sys
import pandas as pd
import gradio as gr
from datetime import datetime

# Add root folder to sys.path to enable clean absolute imports
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.model_loader import load_vlm_model, get_loaded_model_details
from model.report_generator import generate_radiology_report
from utils.validators import validate_image_file
from utils.pdf_generator import generate_radiology_pdf
from utils.report_storage import save_report, get_report_history_dataframe, load_report_by_id

# Initialize output directories
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Try loading the VLM model on startup
print("Initializing AI Radiology Generator backend...")
MODEL, PROCESSOR, IS_MOCK = load_vlm_model()
MODEL_DETAILS = get_loaded_model_details()

def generate_report_workflow(image_path):
    """
    Core pipeline:
    1. Validation
    2. Inference (VLM / Mock)
    3. PDF generation
    4. Local storage
    5. UI refresh
    """
    if not image_path:
        return (
            "", "", "", None, 
            get_report_history_dataframe(), 
            "⚠️ Please upload a chest X-Ray image first."
        )
        
    # Step 1: Validate image
    is_valid, validation_msg = validate_image_file(image_path)
    if not is_valid:
        return (
            "", "", "", None, 
            get_report_history_dataframe(), 
            f"❌ Image Validation Failed: {validation_msg}"
        )
        
    # Step 2: Generate Radiology Report
    try:
        report_data = generate_radiology_report(
            image_path=image_path,
            model=MODEL,
            processor=PROCESSOR,
            is_mock=IS_MOCK
        )
    except Exception as e:
        return (
            "", "", "", None, 
            get_report_history_dataframe(), 
            f"❌ Generation Error: Failed to analyze image. Details: {str(e)}"
        )
        
    findings = report_data["findings"]
    impression = report_data["impression"]
    recommendations = report_data["recommendations"]
    
    # Step 3: Generate PDF
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"Radiology_Report_{timestamp_suffix}.pdf"
    pdf_path = os.path.join(OUTPUTS_DIR, pdf_filename)
    
    try:
        generate_radiology_pdf(
            output_path=pdf_path,
            findings=findings,
            impression=impression,
            recommendations=recommendations,
            image_name=os.path.basename(image_path)
        )
    except Exception as e:
        pdf_path = None
        print(f"Error generating PDF: {str(e)}")
        
    # Step 4: Save JSON data
    try:
        save_report(
            findings=findings,
            impression=impression,
            recommendations=recommendations,
            image_name=image_path,
            pdf_path=pdf_path
        )
    except Exception as e:
        print(f"Error saving report JSON: {str(e)}")
        
    # Step 5: Load updated history
    history_df = get_report_history_dataframe()
    
    status_success = f"✅ Report successfully generated using {MODEL_DETAILS['model_name']}!"
    return findings, impression, recommendations, pdf_path, history_df, status_success

def load_selected_report(evt: gr.SelectData, history_df):
    """
    Triggered when a user clicks on a row in the report history table.
    Reloads that report into the visual fields.
    """
    try:
        row_idx = evt.index[0]
        # Get the Report ID from the first column of selected row
        report_id = history_df.iloc[row_idx]["Report ID"]
        report_data = load_report_by_id(report_id)
        
        if report_data:
            pdf_path = report_data.get("pdf_path")
            # Verify if PDF file still exists
            if pdf_path and not os.path.exists(pdf_path):
                pdf_path = None
                
            status_msg = f"📂 Loaded report: {report_id}."
            return (
                report_data.get("findings", ""),
                report_data.get("impression", ""),
                report_data.get("recommendations", ""),
                pdf_path,
                status_msg
            )
    except Exception as e:
        return "", "", "", None, f"❌ Error loading report: {str(e)}"
        
    return "", "", "", None, "⚠️ Report not found."

def refresh_history():
    return get_report_history_dataframe(), "🔄 History refreshed."

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Force overall PharmEasy clean clinical colors to bypass browser overrides */
body, .gradio-container {
    background-color: #f2f7f6 !important; /* Soft grey-teal background */
    color: #1f2937 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

/* Ensure all text headers and standard paragraphs are dark for high readability */
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4, 
.gradio-container p, .gradio-container li, .gradio-container span:not(.findings-box span, .impression-box span, .recs-box span),
.gradio-container strong, .gradio-container ul, .gradio-container ol {
    color: #1f2937 !important;
}

.gradio-container p.medical-subtitle, .system-status-text {
    color: #4b5563 !important;
}

/* EHR / PharmEasy Navbar */
.ehr-navbar {
    background-color: #10847e !important; /* Trademark PharmEasy Teal */
    color: #ffffff !important;
    padding: 18px 24px;
    border-radius: 12px 12px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(16, 132, 126, 0.15);
}
.ehr-nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.ehr-nav-icon {
    font-size: 1.45rem;
}
.ehr-nav-title {
    font-weight: 800;
    font-size: 1.3rem;
    letter-spacing: -0.01em;
    color: #ffffff !important;
}
.ehr-nav-separator {
    color: #4db3ad;
    font-weight: 300;
}
.ehr-nav-subtitle {
    font-size: 0.95rem;
    color: #b3dfdd !important;
    font-weight: 500;
}
.ehr-nav-status {
    display: flex;
    align-items: center;
    gap: 8px;
    background-color: #0c635f;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}
.ehr-status-dot {
    width: 8px;
    height: 8px;
    background-color: #22c55e;
    border-radius: 9999px;
    display: inline-block;
    box-shadow: 0 0 8px #22c55e;
}
.ehr-status-text {
    color: #e6f4f3 !important;
}

/* Patient Case Sheet Header */
.patient-case-sheet {
    background-color: #ffffff;
    border-bottom: 1px solid #e5ebea;
    border-left: 1px solid #e5ebea;
    border-right: 1px solid #e5ebea;
    padding: 16px 24px;
    border-radius: 0 0 12px 12px;
    margin-bottom: 25px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.case-sheet-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
}
.case-meta-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.case-meta-label {
    font-size: 0.75rem;
    color: #6b7a78 !important;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.case-meta-val {
    font-size: 0.9rem;
    color: #10847e !important; /* Medical teal accent */
    font-weight: 700;
}

.control-panel, .report-panel, .history-panel {
    background-color: #ffffff !important;
    border: 1px solid #e5ebea !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.01) !important;
    margin-bottom: 25px;
}
.control-panel h3, .report-panel h3, .history-panel h3 {
    color: #10847e !important; /* PharmEasy Teal Headers */
    font-weight: 800;
    font-size: 1.15rem !important;
    margin-bottom: 20px !important;
    border-bottom: 2px solid #f2f7f6 !important;
    padding-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.025em;
}
.control-panel p, .report-panel p, .history-panel p,
.control-panel span, .report-panel span, .history-panel span,
.control-panel li, .report-panel li, .control-panel strong, .report-panel strong,
.diagnostic-panel p, .diagnostic-panel li, .diagnostic-panel strong {
    color: #2d3748 !important;
}

.action-btn {
    background: #10847e !important; /* PharmEasy Primary Green Button */
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    box-shadow: 0 4px 12px rgba(16, 132, 126, 0.2) !important;
    transition: all 0.15s ease-in-out !important;
}
.action-btn:hover {
    background: #0c635f !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(16, 132, 126, 0.3) !important;
}
.action-btn:active {
    transform: translateY(0);
}

.diagnostic-panel {
    background-color: #f2f7f6;
    border: 1px solid #e5ebea;
    border-left: 4px solid #10847e;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
}
.diagnostic-panel h4 {
    color: #10847e !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}
.system-status-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #4a5568 !important;
}

.findings-box {
    border-left: 6px solid #10847e !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
    background-color: #f0fbfb !important;
}
.findings-box textarea {
    background-color: #f0fbfb !important;
    color: #0b6b66 !important;
    border: none !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
.findings-box label span {
    color: #0b6b66 !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.impression-box {
    border-left: 6px solid #ef4444 !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
    background-color: #fff5f5 !important;
}
.impression-box textarea {
    background-color: #fff5f5 !important;
    color: #991b1b !important;
    font-weight: 600 !important;
    border: none !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
.impression-box label span {
    color: #991b1b !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.recs-box {
    border-left: 6px solid #10b981 !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
    background-color: #f0fdf4 !important;
}
.recs-box textarea {
    background-color: #f0fdf4 !important;
    color: #065f46 !important;
    border: none !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
.recs-box label span {
    color: #065f46 !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.status-log {
    border-radius: 8px !important;
    background-color: #f2f7f6 !important;
    border: 1px solid #e5ebea !important;
}
.status-log textarea {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #f2f7f6 !important;
    color: #10847e !important;
}

.refresh-btn {
    background-color: #ffffff !important;
    color: #4a5568 !important;
    border: 1px solid #cbd5e0 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.refresh-btn:hover {
    background-color: #f7fafc !important;
}

.history-table {
    border: 1px solid #e5ebea !important;
    background-color: #ffffff !important;
}
.pdf-download-box {
    background-color: #f2f7f6 !important;
    border: 1.5px dashed #10847e !important;
    border-radius: 10px !important;
}
.pdf-download-box, .pdf-download-box * {
    color: #10847e !important;
    background-color: #f2f7f6 !important;
}
.pdf-download-box .file-preview {
    border: none !important;
}
"""

with gr.Blocks(css=custom_css, title="AI Chest X-Ray Radiology Report Generator") as demo:
    
    # 1. Top Clinical Navigation Bar
    gr.HTML(
        """
        <div class="ehr-navbar">
            <div class="ehr-nav-brand">
                <span class="ehr-nav-icon">🏥</span>
                <span class="ehr-nav-title">AURA EHR System</span>
                <span class="ehr-nav-separator">|</span>
                <span class="ehr-nav-subtitle">Radiology Workstation</span>
            </div>
            <div class="ehr-nav-status">
                <span class="ehr-status-dot"></span>
                <span class="ehr-status-text">PACS connected</span>
            </div>
        </div>
        """
    )
    
    # 2. Patient Case Metadata Sheet
    gr.HTML(
        """
        <div class="patient-case-sheet">
            <div class="case-sheet-grid">
                <div class="case-meta-item">
                    <span class="case-meta-label">Patient Identity</span>
                    <span class="case-meta-val">ANONYMOUS / CASE STUDY</span>
                </div>
                <div class="case-meta-item">
                    <span class="case-meta-label">Device Modality</span>
                    <span class="case-meta-val">CR - Computed Radiography</span>
                </div>
                <div class="case-meta-item">
                    <span class="case-meta-label">Exam Study Area</span>
                    <span class="case-meta-val">CHEST (PA VIEW)</span>
                </div>
                <div class="case-meta-item">
                    <span class="case-meta-label">Radiology Standard</span>
                    <span class="case-meta-val">DICOM Level 3 Compliance</span>
                </div>
            </div>
        </div>
        """
    )
    
    # 2. Main Dashboard Layout (Columns)
    with gr.Row():
        
        # Left Column: Upload & Control
        with gr.Column(scale=5, elem_classes=["control-panel"]):
            gr.Markdown("### 📥 1. Chest X-Ray Upload & Controls")
            
            xray_input = gr.Image(
                label="Upload Chest X-Ray (PNG, JPG, JPEG)",
                type="filepath",
                sources=["upload"],
                interactive=True
            )
            
            generate_btn = gr.Button(
                "⚡ Generate Structured Report",
                variant="primary",
                elem_classes=["action-btn"]
            )
            
            status_box = gr.Textbox(
                label="System Logs / Status",
                placeholder="Upload an image and click Generate...",
                interactive=False,
                elem_classes=["status-log"]
            )
            
            # Diagnostic Info
            with gr.Column(elem_classes=["diagnostic-panel"]):
                gr.Markdown("#### 🖥️ AI Engine Status")
                gr.Markdown(
                    f"""
                    * **Active Model:** `{MODEL_DETAILS['model_name']}`
                    * **Compute Device:** `{MODEL_DETAILS['device'].upper()}`
                    * **Operation Mode:** `{'MOCK ENGINE (Offline Fallback)' if MODEL_DETAILS['is_mock'] else 'ACTIVE VLM INFERENCE'}`
                    """,
                    elem_classes=["system-status-text"]
                )
                
        # Right Column: Generated Report Details
        with gr.Column(scale=7, elem_classes=["report-panel"]):
            gr.Markdown("### 📄 2. Generated Radiology Report")
            
            findings_out = gr.Textbox(
                label="FINDINGS",
                lines=6,
                placeholder="Findings will be generated here...",
                interactive=False,
                elem_classes=["findings-box"]
            )
            
            impression_out = gr.Textbox(
                label="IMPRESSION",
                lines=3,
                placeholder="Clinical impression will be generated here...",
                interactive=False,
                elem_classes=["impression-box"]
            )
            
            recs_out = gr.Textbox(
                label="RECOMMENDATIONS",
                lines=4,
                placeholder="Recommendations will be generated here...",
                interactive=False,
                elem_classes=["recs-box"]
            )
                
            pdf_out = gr.File(
                label="📥 Download Exported PDF Report",
                interactive=False,
                elem_classes=["pdf-download-box"]
            )
            
    # 3. Report History Log Section (Bottom)
    gr.Markdown("---")
    with gr.Row(elem_classes=["history-panel"]):
        with gr.Column():
            gr.Markdown("### 🗃️ 3. Report History Database")
            gr.Markdown("*Click on any row in the history list below to view/reload that report.*")
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Database", variant="secondary", size="sm", elem_classes=["refresh-btn"])
                
            history_table = gr.Dataframe(
                value=get_report_history_dataframe(),
                headers=["Report ID", "Timestamp", "Image Name", "Impression"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                elem_classes=["history-table"]
            )
            
    # --- Event Handlers ---
    
    # 1. Click Generate Report
    generate_btn.click(
        fn=generate_report_workflow,
        inputs=[xray_input],
        outputs=[findings_out, impression_out, recs_out, pdf_out, history_table, status_box]
    )
    
    # 2. Click a row in the history list
    history_table.select(
        fn=load_selected_report,
        inputs=[history_table],
        outputs=[findings_out, impression_out, recs_out, pdf_out, status_box]
    )
    
    # 3. Refresh History
    refresh_btn.click(
        fn=refresh_history,
        inputs=[],
        outputs=[history_table, status_box]
    )

# Launch when executed directly
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
