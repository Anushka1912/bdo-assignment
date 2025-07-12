# bdo-assignment

## Overview

This is a Google Cloud Run service for processing financial statement PDFs and evaluating classification accuracy via two HTTP endpoints:

### `/classification`
- Extracts text from each page of a PDF.
- Classifies pages into types like **Balance Sheet**, **Income Statement**, etc.
- Saves classification results as CSV in a **Google Cloud Storage** bucket.

### `/evaluation-metrics`
- Compares classification results with a ground truth CSV.
- Returns evaluation metrics like **accuracy**, **precision**, **recall**, and **F1 score**.

---

## Built With

- **Google Cloud Vision API** – for OCR-based text extraction from PDFs
- **pdfplumber** – for PDF page parsing and table/text extraction
- **scikit-learn** – for computing evaluation metrics
- **Google Cloud Storage (GCS)** – for storing classification and results CSVs

---

## Prerequisites

Before deploying or running locally, make sure the following are set up:

1. A **Google Cloud Project** with billing enabled.
2. The following **APIs enabled**:
   - Cloud Vision API
   - Cloud Storage API
3. A **GCS bucket**
4. A **Service Account** with the following roles:
   - `roles/cloudvision.user`
   - `roles/storage.objectCreator`
5. Download the service account key as `auth.json`
6. **Python 3.9+**

---

## Folder Structure

```bash
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── auth.json             # GCP service account key (add your creds)
├── README.md             # You're here :)
