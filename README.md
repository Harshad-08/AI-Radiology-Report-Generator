# PS-H1: AI-Powered Chest X-Ray Radiology Report Generator

An automated, local-first healthcare AI web application that accepts a Chest X-Ray (CXR) image and generates a structured, print-ready radiology report containing Findings, Impression, and Clinical Recommendations. Built with PyTorch, Hugging Face Transformers, Gradio, and ReportLab.

---

## 1. Project Overview
This clinical decision support application processes Chest X-Ray (PA/Lateral views) images using Medical Vision-Language Models (VLM) to automate clinical documentation. It generates structured findings, clinical impressions, and actionable follow-up recommendations. It also provides immediate local storage logging (JSON database) and exportable, print-ready PDF reports with professional clinical styling.

---

## 2. Problem Statement
In clinical settings, radiologists face heavy workloads, leading to burnout and delays in reporting. Chest X-rays are the most common diagnostic imaging test globally. Automating the initial draft of structured chest radiographs helps triage critical findings, speeds up documentation cycles, and serves as a reliable second-reader decision support tool.

This application is strictly focused on **Chest X-Rays** and implements a multi-layer validator to reject non-CXR inputs (such as MRI, CT, bone fractures, dental X-rays, or random photos) to ensure safety and clinical relevance.

---

## 3. Features
* **Chest X-Ray Image Upload & Visualizer**: Support for PNG, JPG, and JPEG files.
* **Intelligent Clinical Validator**: Automatically checks image integrity, grayscale saturation levels, aspect ratios, and quadrant brightness profiles to ensure only Chest X-rays are processed.
* **Multimodal Local VLM Inference**:
  1. **Primary VLM (MedBLIP)**: Tries to load MedBLIP architectures (e.g. `unni12345/MedBlip2` or `loopback-kr/Ours-MedBLIP-ep3-batch2-len1024`).
  2. **Secondary VLM (Chest-Radiology Fine-tuned)**: Tries fine-tuned chest X-ray captioning models (e.g. `umarigan/blip-image-captioning-base-chestxray-finetuned` or `adibvafa/BLIP-MIMIC-CXR`).
  3. **VLM Fallback**: Salesforce BLIP base VLM (`Salesforce/blip-image-captioning-base`).
  4. **Offline/OOM Fallback**: Highly sophisticated, local pixel-characteristic clinical rules engine.
* **Structured Clinical Report Generation**: Outputs reports strictly adhering to medical standards:
  * **FINDINGS**: Text describing organ system states (lungs, heart, bony structure).
  * **IMPRESSION**: Summary of pathology (e.g., Pneumonia, Cardiomegaly, Atelectasis, normal).
  * **RECOMMENDATIONS**: Actionable clinical follow-up advice.
* **Professional PDF Exporter**: Compiles reports into clinical, print-ready PDFs using a clean medical slate-blue/emerald layout, report headers, dynamic page numbers ("Page X of Y"), and signature fields.
* **Local Report History Database**: Saves all generated reports to local JSON logs and provides an interactive sidebar on the dashboard to reload historical reports instantly.

---

## 4. Architecture Diagram

```
[ Chest X-Ray Image ]
          │
          ▼
[ Image Validator & Heuristics ] ──(Reject Non-CXR or Corrupted)──► [ gr.Warning() Display ]
          │ (Pass)
          ▼
[ Image Preprocessing (384x384, RGB) ]
          │
          ▼
[ Medical Vision-Language Model (MedBLIP / Fallback VLM) ]
          │
          ▼
[ Clinical Findings Generator / Template Mapper ]
          │
          ├─────────────────────────────┐
          ▼                             ▼
   [ Report Formatter ]           [ PDF Exporter ]
          │                             │
          ▼                             ▼
[ Local JSON Database ]        [ Print-Ready PDF ]
(reports/*.json)               (outputs/*.pdf)
```

---

## 5. Folder Structure
```
AI-Radiology-Report-Generator/
├── app.py                     # Main Gradio application entrypoint
├── requirements.txt           # Python application dependencies
├── README.md                  # Project documentation
├── verify_app.py              # Programmatic verification and test suite
│
├── model/
│   ├── model_loader.py        # Safe VLM downloader & device auto-detection
│   ├── image_processor.py     # Image sizing and color channel preprocessor
│   └── report_generator.py    # Text inference, template mapping, and mock engine
│
├── utils/
│   ├── pdf_generator.py       # ReportLab PDF compiler
│   ├── report_storage.py      # JSON reads/writes and history formatter
│   ├── validators.py          # CXR aspect ratio & quadrant brightness checks
│   └── generate_samples.py    # Synthetic image generator for instant testing
│
├── reports/                   # Saved JSON reports database
├── sample_images/             # Programmatically generated test X-rays
├── outputs/                   # Exported PDF documents
└── assets/                    # Static media files
```

---

## 6. Installation Guide

### Prerequisites
* Python 3.9, 3.10, 3.11, or 3.12 (with `pip` and `venv` installed).
* A local environment (Mac OS, Windows, or Linux).

### Step-by-Step Setup
1. **Navigate to Project Directory**:
   ```bash
   cd AI-Radiology-Report-Generator
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate Virtual Environment**:
   * **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```
   * **Windows**:
     ```cmd
     venv\Scripts\activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Generate Synthetic Sample Images** (Optional - if you don't have real CXR files on hand):
   ```bash
   python utils/generate_samples.py
   ```

---

## 7. Usage Instructions

### Run Programmatic Verification Tests
Before launching the server, verify the pipeline:
```bash
python verify_app.py
```

### Launch the Gradio Dashboard
Start the local server:
```bash
python app.py
```

After launch, open your browser and navigate to the local address displayed in the console:
`http://127.0.0.1:7860`

### Operating the Dashboard
1. **Upload X-Ray**: Drag and drop a Chest X-Ray image (e.g., from `sample_images/normal_chest_xray.png` or `sample_images/abnormal_chest_xray.png`) into the upload box.
2. **Generate**: Click **Generate Structured Report**.
3. **Review**: Check the **Findings**, **Impression**, and **Recommendations** fields.
4. **Download**: Use the file component at the bottom right to download the professionally styled PDF.
5. **View History**: Look at the history table at the bottom. Click any row to reload and view past radiology reports.

---

## 8. Technologies Used
* **Python**: Core scripting and logic.
* **PyTorch & Hugging Face Transformers**: VLM loading and inference pipeline.
* **Gradio**: Web interface and interactive UI blocks.
* **ReportLab**: Dynamic clinical PDF document generation.
* **Pillow**: Grayscale conversions, aspect calculations, and image verification.
* **Pandas**: Tabular log database indexing.

---

## 9. Future Enhancements
* **Pathology Heatmapping**: Integrate Grad-CAM/attention visualizations to highlight localized chest lesions directly on the preview image.
* **Multi-View Comparison**: Support comparing current PA view against historical CXR records to generate delta comparisons (e.g., "right lower lobe consolidation has resolved compared to prior exam").
* **Doctor-in-the-Loop Signoff**: Add digital signature verification, allowing clinicians to make modifications before exporting the signed report.

---

## 10. Screenshots Section
*(Add screenshots showing the Gradio layout, image upload, generated text, PDF report layout, and the history table here)*
