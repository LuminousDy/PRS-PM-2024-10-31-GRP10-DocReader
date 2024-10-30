import streamlit as st
import os
from pathlib import Path
import pytesseract
import shutil
from PIL import Image
import io
import time
from demo import (
    convert_files_in_folder,
    process_images,
    extract_text_from_images,
    summarize_text_in_folder,
    process_keywords_in_folder,
    process_wikipedia_search_in_folder,
    get_image_paths_from_folder,
    YOLOv10
)
import base64
import re
from PyPDF2 import PdfMerger

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def create_pdf_from_images(image_paths, output_path):
    images = [Image.open(path).convert('RGB') for path in image_paths]
    images[0].save(output_path, save_all=True, append_images=images[1:])

def main():
    st.set_page_config(page_title="Document Processing App", layout="wide", page_icon="📄")

    set_png_as_page_bg('bgpic.png')
    
    st.title("📄 Intelligent Document Processing System")
    st.subheader("Upload, Analyze, Summarize - One-Stop Document Intelligence Solution")
    
    # Use session_state to save the state of uploaded files and processing results
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'annotated_image_paths' not in st.session_state:
        st.session_state.annotated_image_paths = []
    if 'processing_completed' not in st.session_state:
        st.session_state.processing_completed = False
    if 'processed_dir' not in st.session_state:
        st.session_state.processed_dir = None
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to Use")
    st.sidebar.markdown("""
    1. Upload a PDF or PNG file
    2. Click the 'Start Backend Processing' button
    3. View processing results and summary
    """)
    
    uploaded_file = st.sidebar.file_uploader("Upload file", type=["pdf", "png"])
    
    # If a new file is uploaded, update session_state and reset processing status
    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.processing_completed = False
        st.session_state.annotated_image_paths = []
        st.session_state.processed_dir = None

    if st.session_state.uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'./tesseract/tesseract.exe'
        
        # Clear and recreate temporary folder
        temp_dir = "temp_uploads"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # Clear and recreate processed_images folder only if it's a new upload
        if st.session_state.processed_dir is None:
            processed_dir = f"processed_images_{time.strftime('%Y%m%d%H%M%S')}"
            os.makedirs(processed_dir)
            st.session_state.processed_dir = processed_dir
        else:
            processed_dir = st.session_state.processed_dir
        
        # Save the uploaded file
        file_path = os.path.join(temp_dir, st.session_state.uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())
        
        st.info("File saved, starting processing...")
        
        # Call file processing function
        convert_files_in_folder(temp_dir, processed_dir)
        
        st.success("File processing completed!")
        
        if st.sidebar.button("Start Backend Processing 🚀", key="start_processing"):
            st.session_state.processing_completed = False
            with st.spinner("Processing..."):
                try:
                    start_time = time.time()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # YOLO model processing
                    status_text.info("Performing YOLO object detection...")
                    model = YOLOv10('yolov10x_best.pt')
                    image_paths = get_image_paths_from_folder(processed_dir)
                    process_images(model, image_paths, output_root_dir=processed_dir)
                    progress_bar.progress(20)
                    
                    # Load and save annotated images
                    annotated_dir = os.path.join(processed_dir, "annotated_pages")
                    os.makedirs(annotated_dir, exist_ok=True)
                    file_paths = []
                    for root, dirs, files in os.walk(processed_dir):
                        for file_name in files:
                            if file_name.startswith("annotated_"):
                                full_path = os.path.join(root, file_name)
                                new_path = os.path.join(annotated_dir, file_name)
                                shutil.copy(full_path, new_path)
                                file_paths.append(new_path)
                    
                    # Sort file paths using natural sorting
                    file_paths.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
                    
                    # Store file paths in session state
                    st.session_state.annotated_image_paths = file_paths
                    
                    progress_bar.progress(40)
                    
                    # Text extraction
                    status_text.info("Extracting text...")
                    extract_text_from_images(processed_dir)
                    progress_bar.progress(60)
                    
                    # Generate summary
                    status_text.info("Generating summary...")
                    summarize_text_in_folder(processed_dir)
                    progress_bar.progress(80)
                    
                    # Keyword extraction and Wikipedia search
                    status_text.info("Extracting keywords and retrieving Wikipedia information...")
                    process_keywords_in_folder(processed_dir)
                    process_wikipedia_search_in_folder(processed_dir)
                    progress_bar.progress(100)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    st.success(f"Backend processing completed! Total time: {processing_time:.2f} seconds")
                    st.session_state.processing_completed = True
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
        
        # Only display results after processing is completed
        if st.session_state.processing_completed:
            # Display annotated pages with pagination
            if st.session_state.annotated_image_paths:
                st.subheader("Annotated Pages")
                
                # Create PDF from annotated images
                pdf_path = os.path.join(st.session_state.processed_dir, "annotated_pages.pdf")
                create_pdf_from_images(st.session_state.annotated_image_paths, pdf_path)
                
                # Add download button for PDF
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download Annotated PDF",
                        data=pdf_file,
                        file_name="annotated_pages.pdf",
                        mime="application/pdf"
                    )
                
                page_number = st.number_input("Page", min_value=1, max_value=len(st.session_state.annotated_image_paths), value=1)
                image_path = st.session_state.annotated_image_paths[page_number-1]
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=f"Page {page_number}", use_column_width=True)
                except FileNotFoundError:
                    st.error(f"Image file not found: {image_path}")
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            else:
                st.warning("No annotated images found. Please check the processed_images folder.")
            
            # Display results
            st.header("Processing Results")
            for folder_name in os.listdir(processed_dir):
                folder_path = os.path.join(processed_dir, folder_name)
                if os.path.isdir(folder_path) and folder_name != "annotated_pages":
                    st.subheader(f"Summary for {folder_name}")
                    summary_file = os.path.join(folder_path, f"{folder_name}_summary.txt")
                    keywords_file = os.path.join(folder_path, f"{folder_name}_keywords.txt")
                    wiki_file = os.path.join(folder_path, f"{folder_name}_wikipedia_search_results.txt")
                    
                    if os.path.exists(summary_file) and os.path.exists(keywords_file) and os.path.exists(wiki_file):
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = f.read()
                        with open(keywords_file, 'r', encoding='utf-8') as f:
                            keywords = [line.strip() for line in f if line.strip()]
                        
                        # Add hyperlinks to keywords (exact match with word boundaries)
                        for keyword in keywords:
                            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b')
                            wiki_link = f"[{keyword}](https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')})"
                            summary = pattern.sub(wiki_link, summary)
                        
                        st.markdown(summary, unsafe_allow_html=True)
    else:
        st.title("Welcome to the Document Processing App 👋")
        st.write("Please upload a PDF or PNG file in the sidebar to begin processing.")
    
        st.subheader("User Testimonials")
        st.markdown("""
        > "This app has greatly improved my work efficiency!" - John Doe, Data Analyst
        
        > "Intuitive interface, powerful features. Highly recommended!" - Jane Smith, Project Manager
        """)
        
        st.subheader("Frequently Asked Questions")
        faq_data = [
            ("What file types are supported?", "Currently, we support PDF and PNG files."),
            ("How long does processing take?", "Processing time varies depending on file size and complexity, but typically takes 2-5 minutes."),
            ("Is my data secure?", "Yes, we prioritize data security. All uploaded files are processed locally and deleted after analysis."),
            ("Can I download the results?", "Yes, you can download the annotated document and summary after processing.")
        ]
        for question, answer in faq_data:
            with st.expander(question):
                st.write(answer)

if __name__ == "__main__":
    main()
