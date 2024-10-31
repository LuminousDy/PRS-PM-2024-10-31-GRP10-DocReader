# DocReader: Automated Document Processing and Knowledge Extraction

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/LuminousDy/PRS-PM-2024-10-31-GRP10-DocReader)

## Project Title
DocReader - PRS-PM-2024-10-31-GRP10

## Executive Summary
DocReader is a comprehensive document processing platform designed to automate layout detection, text extraction, summarization, and contextual keyword retrieval. Integrating advanced models such as YOLOv10 for layout detection, Tesseract OCR for text extraction, BART for summarization, and DistilBERT for keyword extraction, the system enhances document comprehension and accessibility. Aiming to streamline document management in academia, business, and legal domains, DocReader also enriches extracted data by linking keywords to relevant Wikipedia information, offering contextual background for users to quickly grasp essential document insights.

## Project Objectives
DocReader’s primary goal is to streamline document processing by automating extraction and knowledge retrieval. Key objectives include:
1. **Automated Document Structure Detection**: Identify document components like headers, paragraphs, and tables for accurate data extraction.
2. **High-Accuracy Text Recognition**: Utilize OCR for clear and editable text conversion, even in complex layouts.
3. **Concise Summarization**: Generate brief yet comprehensive summaries for dense documents.
4. **Contextual Keyword Extraction**: Highlight core concepts and terms with linked contextual information from Wikipedia.

## System Architecture
The architecture of DocReader consists of a single frontend component for user interaction:

1. **User Interface (UI)**: A Streamlit-based dashboard that allows for document upload, process initiation, and results viewing. It handles document processing requests and displays the extracted content.

## Project Directory Structure

Based on the provided structure, the main directories and their purposes are:

- **Miscellaneous/**: Contains additional resources or miscellaneous files.
- **ProjectReport/**: Documentation and reports related to the project's development and findings.
- **SystemCode/**: The core codebase, which is organized as follows:
  - **images_folder/**: Stores image files generated during document analysis, further divided by document and page number.
  - **EconAgent/**: Example processed images for demonstration, organized by page number and containing multiple detected elements.
  - **pdf_folder/**: Contains original PDF files uploaded for processing.
  - **poppler-24.08.0/**: Poppler utility files for handling PDF operations and rendering.
  - **processed_images/**: Stores images of processed documents post-segmentation.
  - **temp_uploads/**: Temporary storage for files uploaded via the UI.
  - **tesseract/**: Contains Tesseract OCR data and configuration files.
  - **__pycache__/**: Cached Python files for quicker execution.
- **Video/**: Contains video demonstrations of the project’s functionalities.

## Core Modules

### 1. Layout Detection Module (YOLOv10)
   - Uses YOLOv10 to detect and organize document components, including headings, tables, and paragraphs, allowing for structured information extraction.

### 2. Text Recognition Module (Tesseract OCR)
   - Employs Tesseract OCR to transform detected segments into editable text, ensuring accessibility for scanned and non-digital documents.

### 3. Summarization Module (BART)
   - BART model generates concise, accurate summaries that capture essential information, streamlining document review.

### 4. Keyword Extraction and Contextualization Module (DistilBERT & Wikipedia API)
   - Extracts contextually significant keywords and connects them to Wikipedia, providing relevant background information directly in the system.

## Installation

To set up the DocReader project, follow these steps:

1. **Clone the repository**:
git clone https://github.com/LuminousDy/PRS-PM-2024-10-31-GRP10-DocReader.git
2. **Navigate to the project directory**:
cd PRS-PM-2024-10-31-GRP10-DocReader
3. **Install dependencies**:
pip install -r requirements.txt # For Python dependencies

Ensure you have **Poppler** installed for PDF processing and **Tesseract** for OCR functionalities, as the paths to `poppler-24.08.0` and `tesseract` directories indicate these dependencies are crucial for document processing.

## Usage

To run the project, follow these steps:

1. **Launch the Frontend Interface**:
streamlit run homepage.py
This command will start the Streamlit application for the frontend interface, allowing interaction with the document processing pipeline.

2. **Access the Web Interface**:
- Open a web browser and navigate to the displayed Streamlit URL (typically `http://localhost:8501`) to upload documents, initiate processing, and view results.

### Execution Workflow:
- **Document Upload**: Upload a document (PDF or image format) via the Streamlit web interface, which is temporarily stored in `temp_uploads/`.
- **Document Processing**: The system runs layout detection on each page, storing segmented images in `images_folder/` and structured data in `processed_images/`.
- **Result Retrieval**: Summarization and keywords are generated, linked to Wikipedia, and made accessible on the results page.

## Solution Implementation
- **Modular System Design**: Each module operates independently, supporting the complete document analysis workflow from upload to result display.
- **Integrated Knowledge Retrieval**: Wikipedia API adds contextual depth by linking keywords to background information.

## Future Plans
1. **Enhance OCR and Layout Detection**: Improve recognition for low-resolution or complexly structured documents.
2. **Expand Knowledge Sources**: Integrate academic and specialized databases to enrich keyword context.
3. **Optimize Performance and Scalability**: Adapt the architecture for broader applications across diverse document-intensive industries.

## Contributors
- **Ding Yi (A0295756J)**
- **Liu Lihao (A0296992A)**
- **Lou Shengxin (A0397330A)**
- **Shi Haocheng (A0296265R)**
- **Yang Runzhi (A0297296H)**

## License
This project is licensed under the Apache-2.0 License.
