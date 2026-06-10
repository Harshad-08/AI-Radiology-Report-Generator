import os
import sys

# Add root folder to sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.validators import validate_image_file
from utils.pdf_generator import generate_radiology_pdf
from utils.report_storage import save_report, get_report_history, get_report_history_dataframe
from model.model_loader import load_vlm_model
from model.report_generator import generate_radiology_report

def run_tests():
    print("==================================================")
    print("STARTING PROGRAMMATIC PIPELINE VERIFICATION")
    print("==================================================\n")
    
    # Paths setup
    sample_dir = os.path.join(ROOT_DIR, "sample_images")
    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    normal_img = os.path.join(sample_dir, "normal_chest_xray.png")
    abnormal_img = os.path.join(sample_dir, "abnormal_chest_xray.png")
    
    # Test 1: Validate generated synthetic images
    print("Test 1: Validating synthetic Chest X-Ray images...")
    for img_path in [normal_img, abnormal_img]:
        is_ok, msg = validate_image_file(img_path)
        print(f"  Image: {os.path.basename(img_path)}")
        print(f"  Valid: {is_ok} | Msg: {msg}")
        assert is_ok, f"Verification failed: Synthetic CXR {img_path} was marked as invalid!"
    print("✔️ Test 1 Passed.\n")
    
    # Test 2: Rejecting non-image file
    print("Test 2: Validating non-image/corrupted file handling...")
    bad_file = os.path.join(outputs_dir, "invalid_test.png")
    with open(bad_file, "w") as f:
        f.write("This is a mock text file, not a chest x-ray.")
        
    is_ok, msg = validate_image_file(bad_file)
    print(f"  Bad File: {os.path.basename(bad_file)}")
    print(f"  Valid: {is_ok} | Msg: {msg}")
    assert not is_ok, "Verification failed: Bad file was accepted!"
    os.remove(bad_file)
    print("✔️ Test 2 Passed.\n")
    
    # Test 3: Load VLM / Mock engine
    print("Test 3: Testing Model Loader fallback/mock engine...")
    # Force Mock model for rapid programmatic test to avoid large network download
    os.environ["FORCE_MOCK_MODEL"] = "true"
    model, processor, is_mock = load_vlm_model()
    print(f"  Model Loaded: {model}")
    print(f"  Processor Loaded: {processor}")
    print(f"  Is Mock Mode: {is_mock}")
    assert is_mock, "Verification failed: Force mock was ignored!"
    print("✔️ Test 3 Passed.\n")
    
    # Test 4: Report generation
    print("Test 4: Running report generation...")
    report_normal = generate_radiology_report(normal_img, model, processor, is_mock)
    report_abnormal = generate_radiology_report(abnormal_img, model, processor, is_mock)
    
    print("  Normal CXR Report Output:")
    print(f"    Findings: {report_normal['findings'][:50]}...")
    print(f"    Impression: {report_normal['impression']}")
    
    print("  Abnormal CXR Report Output:")
    print(f"    Findings: {report_abnormal['findings'][:50]}...")
    print(f"    Impression: {report_abnormal['impression']}")
    
    assert "findings" in report_normal
    assert "impression" in report_normal
    assert "recommendations" in report_normal
    print("✔️ Test 4 Passed.\n")
    
    # Test 5: PDF Export
    print("Test 5: Exporting report to PDF...")
    pdf_path = os.path.join(outputs_dir, "test_report.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        
    generate_radiology_pdf(
        pdf_path,
        report_normal["findings"],
        report_normal["impression"],
        report_normal["recommendations"],
        "normal_chest_xray.png"
    )
    
    print(f"  PDF output path: {pdf_path}")
    print(f"  PDF exists: {os.path.exists(pdf_path)}")
    assert os.path.exists(pdf_path), "Verification failed: PDF report was not created!"
    print("✔️ Test 5 Passed.\n")
    
    # Test 6: Report Storage (JSON logs)
    print("Test 6: Saving and retrieving report in JSON storage...")
    json_path = save_report(
        findings=report_normal["findings"],
        impression=report_normal["impression"],
        recommendations=report_normal["recommendations"],
        image_name=normal_img,
        pdf_path=pdf_path
    )
    print(f"  JSON output path: {json_path}")
    assert os.path.exists(json_path), "Verification failed: JSON report was not saved!"
    
    history_df = get_report_history_dataframe()
    print(f"  History size: {len(history_df)}")
    assert len(history_df) >= 1, "Verification failed: History dataframe is empty!"
    print("✔️ Test 6 Passed.\n")
    
    print("==================================================")
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
