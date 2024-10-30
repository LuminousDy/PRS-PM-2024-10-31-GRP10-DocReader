# pip install -q git+https://github.com/THU-MIG/yolov10.git
# pip install -q supervision
# wget https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt
# wget https://raw.githubusercontent.com/moured/YOLOv10-Document-Layout-Analysis/main/images/input_sample.png
# pip install pdf2image
# pip install pillow
# pip install python-docx
# pip install comtypes 
# pip install pdf2image]
# pip install nltk
# import nltk
# nltk.download('stopwords')

import supervision as sv
from ultralytics import YOLOv10
import os
from pdf2image import convert_from_path
import comtypes.client  
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import nltk
from nltk.corpus import stopwords
from torch.nn.functional import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import re
import requests
import numpy as np
import random

# --- Utility Functions ---

def remove_html_tags_and_cleanup(text):
    clean_text = re.sub('<.*?>', '', text)
    clean_text = re.sub(r'\n+', '\n', clean_text)
    return clean_text

def pdf_to_images(pdf_path, output_folder):
    file_name = Path(pdf_path).stem  
    file_output_folder = os.path.join(output_folder, file_name)  

    if not os.path.exists(file_output_folder):
        os.makedirs(file_output_folder)

    images = convert_from_path(pdf_path, poppler_path=r'./poppler-24.08.0/Library/bin')
    for i, image in enumerate(images):
        page_folder = os.path.join(file_output_folder, f"page_{i+1}")  
        if not os.path.exists(page_folder):
            os.makedirs(page_folder)

        image_path = os.path.join(page_folder, f"page_{i+1}.png")
        image.save(image_path, "PNG")
        print(f"Page {i+1} of {pdf_path} saved as {image_path}")

def word_to_pdf(docx_path, pdf_path):
    wdFormatPDF = 17
    word = comtypes.client.CreateObject('Word.Application')
    doc = word.Documents.Open(docx_path)
    doc.SaveAs(pdf_path, FileFormat=wdFormatPDF)
    doc.Close()
    word.Quit()

def word_to_images(docx_path, output_folder):
    pdf_path = os.path.splitext(docx_path)[0] + ".pdf"
    word_to_pdf(docx_path, pdf_path)
    pdf_to_images(pdf_path, output_folder)

def image_to_png(image_path, output_folder):
    image = Image.open(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    file_name = Path(image_path).stem
    file_output_folder = os.path.join(output_folder, file_name)

    if not os.path.exists(file_output_folder):
        os.makedirs(file_output_folder)

    page_folder = os.path.join(file_output_folder, "page_1")
    if not os.path.exists(page_folder):
        os.makedirs(page_folder)

    output_path = os.path.join(page_folder, f"{file_name}.png")
    image.save(output_path, "PNG")
    print(f"Image {file_name} saved as {output_path}")

def convert_files_in_folder(folder_path, output_folder):
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = Path(file_path).suffix.lower()

            if file_extension == ".pdf":
                print(f"Detected PDF file: {file_path}")
                pdf_to_images(file_path, output_folder)
            elif file_extension in [".docx", ".doc"]:
                print(f"Detected Word file: {file_path}")
                word_to_images(file_path, output_folder)
            elif file_extension in [".png", ".jpg", ".jpeg"]:
                print(f"Detected image file: {file_path}")
                image_to_png(file_path, output_folder)
            else:
                print(f"Unsupported file type: {file_extension}. Skipping {file_path}")

# --- Image Processing ---

def get_image_paths_from_folder(folder, extensions=['.png', '.jpg', '.jpeg']):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def determine_layout(bounding_boxes):
    x_coordinates = [box[0] for box in bounding_boxes]
    median_x = np.median(x_coordinates)
    left_column = [box for box in bounding_boxes if box[0] < median_x]
    right_column = [box for box in bounding_boxes if box[0] >= median_x]
    
    if len(left_column) > 0 and len(right_column) > 0:
        return "double_column"
    else:
        return "single_column"

def sort_boxes(bounding_boxes, layout):
    if layout == "single_column":
        return sorted(bounding_boxes, key=lambda box: box[1])
    else:
        median_x = np.median([box[0] for box in bounding_boxes])
        left_column = sorted([box for box in bounding_boxes if box[0] < median_x], key=lambda box: box[1])
        right_column = sorted([box for box in bounding_boxes if box[0] >= median_x], key=lambda box: box[1])
        return left_column + right_column

def process_images(model, image_paths, output_root_dir, conf_threshold=0.2, iou_threshold=0.8):
    class_names = model.names
    colors = {class_id: (0, 100, 0) if class_names[class_id] == 'Text' else tuple(random.randint(0, 255) for _ in range(3)) for class_id in class_names}

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    for image_path in image_paths:
        image = Image.open(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        results = model(source=image_path, conf=conf_threshold, iou=iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)

        bounding_boxes = detections.xyxy
        labels = detections.class_id

        layout = determine_layout(bounding_boxes)
        sorted_boxes = sort_boxes(bounding_boxes, layout)

        cropped_images = []
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        file_root_dir = os.path.dirname(image_path)

        for idx, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image.crop((x1, y1, x2, y2))

            original_idx = np.where((bounding_boxes == box).all(axis=1))[0][0]
            label_id = labels[original_idx]
            label_name = class_names[label_id]

            label_dir = os.path.join(file_root_dir, label_name)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            cropped_image_path = os.path.join(label_dir, f"cropped_{label_name}_{idx}.png")
            cropped_image.save(cropped_image_path)
            cropped_images.append(cropped_image)

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 20)
        for box, label_id in zip(bounding_boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            label_name = class_names[label_id]
            color = colors[label_id]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text_bbox = draw.textbbox((x1, y1), label_name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color)
            draw.text((x1 + 2, y1 - text_height - 3), label_name, fill="white", font=font)

        annotated_image_path = os.path.join(file_root_dir, f"annotated_{file_name}.png")
        image.save(annotated_image_path)
        print(f"Saved annotated image as '{annotated_image_path}'")

        # Save layout information
        layout_info_path = os.path.join(file_root_dir, f"{file_name}_layout.txt")
        with open(layout_info_path, 'w') as f:
            f.write(f"Layout: {layout}\n")
        print(f"Saved layout information as '{layout_info_path}'")

# --- Text Extraction ---

def extract_text_from_images(base_folder):
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            all_text = ""

            page_folders = sorted([f for f in os.listdir(folder_path) if f.startswith('page_')])
            for page_folder in page_folders:
                page_folder_path = os.path.join(folder_path, page_folder)
                if os.path.isdir(page_folder_path):
                    text_folder = os.path.join(page_folder_path, 'Text')
                    if os.path.exists(text_folder) and os.path.isdir(text_folder):
                        # Read layout information
                        layout_info_path = os.path.join(page_folder_path, f"{page_folder}_layout.txt")
                        layout = "single_column"  # Default to single column
                        if os.path.exists(layout_info_path):
                            with open(layout_info_path, 'r') as f:
                                layout = f.read().strip().split(': ')[1]

                        # Process images based on layout
                        image_files = sorted(os.listdir(text_folder))
                        if layout == "double_column":
                            left_column = []
                            right_column = []
                            for image_file in image_files:
                                image_path = os.path.join(text_folder, image_file)
                                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    try:
                                        img = Image.open(image_path)
                                        text = pytesseract.image_to_string(img)
                                        clean_text = text.replace('\n', ' ').replace('- ', '')
                                        if img.width < img.height / 2:  # Assuming left column
                                            left_column.append(clean_text)
                                        else:
                                            right_column.append(clean_text)
                                    except Exception as e:
                                        print(f"Error processing {image_path}: {e}")
                            all_text += " ".join(left_column) + " " + " ".join(right_column) + " "
                        else:
                            for image_file in image_files:
                                image_path = os.path.join(text_folder, image_file)
                                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    try:
                                        img = Image.open(image_path)
                                        text = pytesseract.image_to_string(img)
                                        clean_text = text.replace('\n', ' ').replace('- ', '')
                                        all_text += clean_text + " "
                                    except Exception as e:
                                        print(f"Error processing {image_path}: {e}")

            if all_text:
                output_txt_path = os.path.join(folder_path, f"{folder_name}_text_output.txt")
                with open(output_txt_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(all_text.strip())
                print(f"Saved text file: {output_txt_path}")
            else:
                print(f"No text found for folder: {folder_name}")

# --- Summarization ---

def summarize_large_text(text, model_name="facebook/bart-large-cnn", chunk_size=1024, summary_length=1024):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    chunks = []
    current_chunk = []
    tokens_count = 0

    for sentence in text.split('. '): 
        sentence_tokens = tokenizer.tokenize(sentence)
        tokens_count += len(sentence_tokens)

        if tokens_count > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            tokens_count = len(sentence_tokens)

        current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    all_summaries = []
    for chunk in chunks:
        inputs = tokenizer([chunk], max_length=chunk_size, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=summary_length, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        all_summaries.append(summary)

    final_summary = " ".join(all_summaries)
    return final_summary

def summarize_text_in_folder(base_folder, model_name="facebook/bart-large-cnn", chunk_size=1024, summary_length=1024):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            txt_file_name = f"{folder_name}_text_output.txt"
            txt_file_path = os.path.join(folder_path, txt_file_name)
            
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                summary = summarize_large_text(text, model_name, chunk_size, summary_length)
                summary_file_path = os.path.join(folder_path, f"{folder_name}_summary.txt")
                with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(summary)
                print(f"Saved summary for {folder_name} as {summary_file_path}")
            else:
                print(f"No text file found for folder: {folder_name}")

# --- Keyword Extraction ---

def extract_keywords_distilbert_inspec(sentence, tokenizer, model, device):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_ids = torch.argmax(logits, dim=2)

    predicted_keywords = []
    words = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).cpu())

    current_keyword = []
    for word, label_id in zip(words, predicted_class_ids[0].cpu()):
        label = model.config.id2label[label_id.item()]
        if label == 'B-KEY':
            if current_keyword:
                predicted_keywords.append(" ".join(current_keyword))
                current_keyword = []
            current_keyword.append(word)
        elif label == 'I-KEY':
            current_keyword.append(word)
        else:
            if current_keyword:
                predicted_keywords.append(" ".join(current_keyword))

    if current_keyword:
        predicted_keywords.append(" ".join(current_keyword))

    cleaned_keywords = []
    for keyword in predicted_keywords:
        cleaned_keyword = keyword.replace(' ##', '').strip()
        cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword) 
        cleaned_keyword = re.sub(r'[^\w\s]', '', cleaned_keyword)
        if cleaned_keyword: 
            cleaned_keywords.append(cleaned_keyword)

    unique_keywords = list(set(cleaned_keywords))
    return unique_keywords

def process_keywords_in_folder(base_folder):
    tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec")
    model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            summary_file_name = f"{folder_name}_summary.txt"
            summary_file_path = os.path.join(folder_path, summary_file_name)
            
            if os.path.exists(summary_file_path):
                with open(summary_file_path, 'r', encoding='utf-8') as file:
                    summary_text = file.read()
                keywords = extract_keywords_distilbert_inspec(summary_text, tokenizer, model, device)
                keywords_file_path = os.path.join(folder_path, f"{folder_name}_keywords.txt")
                with open(keywords_file_path, 'w', encoding='utf-8') as keyword_file:
                    for keyword in keywords:
                        keyword_file.write(keyword + '\n')
                print(f"Saved keywords to {keywords_file_path}")
            else:
                print(f"No summary file found for folder: {folder_name}")

# --- Wikipedia Search ---

def search_wikipedia(keywords):
    url = "https://en.wikipedia.org/w/api.php"
    search_results = {}

    for keyword in keywords:
        print(f"Searching for: {keyword}")
        params = {
            "action": "query",
            "list": "search",
            "srsearch": keyword,
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            search_results_data = data.get('query', {}).get('search', [])
            if search_results_data:
                result = search_results_data[0]
                title = result['title']
                extract_params = {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "titles": title,
                    "format": "json"
                }
                extract_response = requests.get(url, params=extract_params)
                if extract_response.status_code == 200:
                    extract_data = extract_response.json()
                    pages = extract_data.get('query', {}).get('pages', {})
                    page = next(iter(pages.values()))
                    extract = page.get('extract', '')
                    clean_extract = remove_html_tags_and_cleanup(extract)
                    search_results[keyword] = {"title": title, "extract": clean_extract}
                else:
                    print(f"Failed to fetch extract for '{title}'")
            else:
                print(f"No results found for '{keyword}'.\n")
        else:
            print(f"Failed to fetch data from Wikipedia API for keyword '{keyword}', Status Code: {response.status_code}")
    
    return search_results

def process_wikipedia_search_in_folder(base_folder):
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            keywords_file_name = f"{folder_name}_keywords.txt"
            keywords_file_path = os.path.join(folder_path, keywords_file_name)
            
            if os.path.exists(keywords_file_path):
                with open(keywords_file_path, 'r', encoding='utf-8') as file:
                    keywords = [line.strip() for line in file if line.strip()]
                if keywords:
                    search_results = search_wikipedia(keywords)
                    output_file_path = os.path.join(folder_path, f"{folder_name}_wikipedia_search_results.txt")
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        for keyword, result in search_results.items():
                            output_file.write(f"Keyword: {keyword}\n")
                            output_file.write(f"Title: {result['title']}\n")
                            output_file.write(f"Extract: {result['extract']}\n\n")
                    print(f"Saved Wikipedia search results to {output_file_path}")
            else:
                print(f"No keywords file found for folder: {folder_name}")

# --- Main Function ---

def main():
    # Paths and folder setup
    file_to_convert = "pdf_folder"
    output_folder = "images_folder"
    base_folder = "images_folder"
    pytesseract.pytesseract.tesseract_cmd = r'./tesseract/tesseract.exe'
    
    # Step 1: Convert PDF and Word files to images
    convert_files_in_folder(file_to_convert, output_folder)

    # Step 2: Perform object detection on images
    model = YOLOv10('yolov10x_best.pt')
    image_paths = get_image_paths_from_folder(output_folder)
    process_images(model, image_paths, output_root_dir=output_folder)

    # Step 3: Extract text from images
    extract_text_from_images(base_folder)

    # Step 4: Summarize extracted text
    summarize_text_in_folder(base_folder)

    # Step 5: Extract keywords using DistilBERT
    process_keywords_in_folder(base_folder)

    # Step 6: Search Wikipedia based on extracted keywords
    process_wikipedia_search_in_folder(base_folder)

if __name__ == "__main__":
    main()

