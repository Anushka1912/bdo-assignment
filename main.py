import functions_framework
import json
import os
import re
import csv
import io
import tempfile
import logging
from typing import Dict, List, Any
from google.cloud import vision_v1, storage
from google.cloud.vision_v1 import types
import fitz
import pdfplumber
from PIL import Image, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import fuzz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Extracting numbers from text using regex
def parse_numbers(text: str) -> List[Any]:
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return [float(num) if '.' in num else int(num) for num in numbers]

# Processing a single PDF page with preprocessing, OCR, and classification
def handle_page(page_idx: int, pdf_file: str) -> Dict[str, Any]:
    try:
        page_info = {
            "page_id": page_idx + 1,
            "content": "",
            "data_tables": [],
            "numbers_in_tables": [],
            "ocr_texts": [],
            "doc_type": "Unknown",
            "doc_category": "Standalone",
            "has_table": False,
            "table_contains_numbers": False,
            "statement_type": "Unknown"
        }

        # Initialize Google Cloud Vision client
        ocr_client = vision_v1.ImageAnnotatorClient()

        # Using pdfplumber for text and table extraction
        with pdfplumber.open(pdf_file) as pdf:
            pdf_page = pdf.pages[page_idx]
            page_info["content"] = pdf_page.extract_text() or ""

            # Extract tables with pdfplumber
            tables = pdf_page.extract_tables()
            for table in tables:
                table_info = {
                    "rows": table,
                    "has_numbers": False,
                    "numbers": []
                }
                for row in table:
                    for cell in row:
                        if cell and isinstance(cell, str):
                            numbers = parse_numbers(cell)
                            if numbers:
                                table_info["has_numbers"] = True
                                table_info["numbers"].extend(numbers)
                page_info["data_tables"].append(table_info)
                if table_info["has_numbers"]:
                    page_info["numbers_in_tables"].extend(table_info["numbers"])
                    page_info["has_table"] = True
                    page_info["table_contains_numbers"] = True

        # Converting PDF page to image and preprocessing
        pdf_document = fitz.open(pdf_file)
        pdf_page = pdf_document[page_idx]
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
        img = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert('L')  # Convert to grayscale
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(2.0)  # Boost contrast
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(2.0)  # Boost sharpness
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        content = img_buffer.getvalue()
        pdf_document.close()

        # Performing OCR with Google Cloud Vision API
        image = types.Image(content=content)
        ocr_response = ocr_client.document_text_detection(image=image)
        ocr_text = ocr_response.full_text_annotation.text if ocr_response.full_text_annotation else ""

        if ocr_text:
            page_info["ocr_texts"].append({
                "image_id": page_idx,
                "type": "pdf_page_image",
                "text": ocr_text
            })

            # Detecting table-like structures in OCR text
            lines = ocr_text.split('\n')
            table_info = {
                "rows": [],
                "has_numbers": False,
                "numbers": []
            }
            current_row = []
            for line in lines:
                if line.strip():
                    current_row.append(line.strip())
                    numbers = parse_numbers(line)
                    if numbers:
                        table_info["has_numbers"] = True
                        table_info["numbers"].extend(numbers)
                else:
                    if current_row:
                        table_info["rows"].append(current_row)
                        current_row = []
            if current_row:
                table_info["rows"].append(current_row)

            if table_info["rows"]:
                page_info["data_tables"].append(table_info)
                if table_info["has_numbers"]:
                    page_info["numbers_in_tables"].extend(table_info["numbers"])
                    page_info["has_table"] = True
                    page_info["table_contains_numbers"] = True

        # Classifying document type and category
        combined_content = page_info["content"] + "\n" + ocr_text
        doc_types = {
            "Balance Sheet": ["balance sheet", "statement of financial position"],
            "Income Statement": ["profit and loss", "income statement", "statement of operations"],
            "Cash Flow Statement": ["cash flow", "statement of cash flows"]
        }
        highest_score = 0
        matched_type = "Unknown"
        for doc_type, patterns in doc_types.items():
            for pattern in patterns:
                score = fuzz.partial_ratio(pattern, combined_content.lower())
                log.debug(f"Matching score for '{pattern}' = {score}")
                if score > 70 and score > highest_score:
                    highest_score = score
                    matched_type = doc_type
        page_info["doc_type"] = matched_type
        page_info["statement_type"] = matched_type

        score = fuzz.partial_ratio("consolidated", combined_content.lower())
        page_info["doc_category"] = "Consolidated" if score > 70 else "Standalone"
        log.debug(f"Consolidated detection score = {score}, result = {page_info['doc_category']}")

        log.debug(f"Handled page {page_idx + 1}: doc_type={page_info['doc_type']}, "
                  f"category={page_info['doc_category']}, has_table={page_info['has_table']}")

        return page_info

    except Exception as e:
        log.error(f"Error handling page {page_idx + 1}: {str(e)}")
        return page_info

# Saving files to Google Cloud Storage
def save_to_bucket(bucket_name: str, file_name: str, file_content: bytes):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(file_content)
        log.info(f"File {file_name} uploaded to bucket {bucket_name}")
    except Exception as e:
        log.error(f"Error uploading {file_name} to bucket {bucket_name}: {str(e)}")

# Loading ground truth data from CSV
def load_ground_truth(ground_truth_file: io.BufferedReader) -> List[Dict]:
    try:
        ground_truth_file.seek(0)  # Reset file pointer
        reader = csv.DictReader(io.TextIOWrapper(ground_truth_file, encoding='utf-8'))
        return [
            {
                "page_id": int(row["page_id"]),
                "statement_type": row.get("doc_type", "None")
            }
            for row in reader
        ]
    except Exception as e:
        log.error(f"Failed to load ground truth: {str(e)}")
        return []

# Evaluating performance metrics
def evaluate_performance(classification_file: io.BufferedReader, ground_truth_file: io.BufferedReader) -> Dict:
    try:
        # Load classification results
        classification_file.seek(0)  # Reset file pointer
        reader = csv.DictReader(io.TextIOWrapper(classification_file, encoding='utf-8'))
        results = [
            {
                "page_id": int(row["page_id"]),
                "statement_type": row.get("doc_type", "None")
            }
            for row in reader
        ]

        # Loading ground truth
        ground_truth = load_ground_truth(ground_truth_file)

        if not ground_truth or not results:
            log.warning("No ground truth or classification data available for evaluation")
            return {"message": "Evaluation not possible"}

        # Aligning by page_id
        true_labels = []
        pred_labels = []
        for gt in ground_truth:
            for res in results:
                if gt["page_id"] == res["page_id"]:
                    true_labels.append(gt["statement_type"])
                    pred_labels.append(res["statement_type"])
                    break

        if not true_labels:
            log.warning("No matching pages found for evaluation")
            return {"message": "No matching pages for evaluation"}

        # Calculating metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        log.info(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        log.error(f"Error evaluating performance: {str(e)}")
        return {"message": f"Evaluation failed: {str(e)}"}

# Single entry point with routing
@functions_framework.http
def main(request):
    # Setting Google Cloud credentials
    credentials_path = "auth.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    # Getting the request path
    path = request.path

    if path == "/classification":
        try:
            # Checking if a file is uploaded
            if 'file' not in request.files:
                log.error("No file provided in the request")
                return "No file provided", 400

            file = request.files['file']
            if not file.filename.lower().endswith('.pdf'):
                log.error("Uploaded file is not a PDF")
                return "File must be a PDF", 400

            # Saving uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                file.save(temp_file.name)
                pdf_file_path = temp_file.name

            result = {
                "doc_name": file.filename,
                "pages": []
            }
            classification_result = {
                "doc_name": file.filename,
                "pages": []
            }

            # Getting number of pages
            pdf_document = fitz.open(pdf_file_path)
            page_count = len(pdf_document)
            pdf_document.close()

            # Processing pages concurrently
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(handle_page, i, pdf_file_path) for i in range(page_count)]
                for future in futures:
                    page_info = future.result()
                    result["pages"].append(page_info)
                    classification_result["pages"].append({
                        "page_id": page_info["page_id"],
                        "doc_type": page_info["doc_type"],
                        "doc_category": page_info["doc_category"],
                        "has_table": page_info["has_table"],
                        "table_contains_numbers": page_info["table_contains_numbers"]
                    })

            # Saving extracted data to JSON
            output_filename = f"{os.path.splitext(file.filename)[0]}_processed.json"
            json_content = json.dumps(result, indent=2, ensure_ascii=False)
            save_to_bucket("files-bdo", output_filename, json_content.encode('utf-8'))

            # Saving classification results to CSV
            class_filename = f"{os.path.splitext(file.filename)[0]}_classification.csv"
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["doc_name", "page_id", "doc_type", "doc_category", "has_table", "table_contains_numbers"])
            writer.writeheader()
            for page in classification_result["pages"]:
                writer.writerow({
                    "doc_name": classification_result["doc_name"],
                    "page_id": page["page_id"],
                    "doc_type": page["doc_type"],
                    "doc_category": page["doc_category"],
                    "has_table": page["has_table"],
                    "table_contains_numbers": page["table_contains_numbers"]
                })
            csv_content = csv_buffer.getvalue()
            save_to_bucket("files-bdo", class_filename, csv_content.encode('utf-8'))

            # Cleaning up temporary file
            os.unlink(pdf_file_path)

            # Returning CSV content as response
            return csv_content, 200, {'Content-Type': 'text/csv'}

        except Exception as e:
            log.error(f"Error processing classification request: {str(e)}")
            return f"Error processing PDF: {str(e)}", 500

    elif path == "/evaluation-metrics":
        try:
            # Checking if both files are uploaded
            if 'classification' not in request.files or 'ground_truth' not in request.files:
                log.error("Missing classification or ground truth file")
                return json.dumps({"message": "Both classification and ground truth CSV files are required"}), 400

            classification_file = request.files['classification']
            ground_truth_file = request.files['ground_truth']

            if not classification_file.filename.lower().endswith('.csv') or not ground_truth_file.filename.lower().endswith('.csv'):
                log.error("Both files must be CSV")
                return json.dumps({"message": "Both files must be CSV"}), 400

            # Evaluating performance
            metrics = evaluate_performance(classification_file, ground_truth_file)
            return json.dumps(metrics, indent=2), 200, {'Content-Type': 'application/json'}

        except Exception as e:
            log.error(f"Error processing evaluation request: {str(e)}")
            return json.dumps({"message": f"Error evaluating metrics: {str(e)}"}), 500

    else:
        log.error(f"Invalid endpoint: {path}")
        return json.dumps({"message": "Invalid endpoint. Use /classification or /evaluation-metrics"}), 404
