# pdf_processor.py

import io
import re
from typing import Optional, Dict
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

class PDFProcessor:
    """Handles PDF file processing and text extraction."""
    
    def __init__(self):
        self.supported_languages = ['eng']
        
    def extract_text_from_image(self, image_data: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang='eng')
            return text
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""
    
    def extract_text_from_pdf(self, pdf_file, password: Optional[str] = None) -> Dict[str, any]:
        try:
            pdf_bytes = pdf_file.read()
            file_size = len(pdf_bytes)
            try:
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                if pdf_document.needs_pass:
                    if password:
                        if not pdf_document.authenticate(password):
                            return {"error": "Invalid password"}
                    else:
                        return {"error": "PDF is encrypted, password required"}
            except Exception as e:
                return {"error": f"Error opening PDF: {str(e)}"}
            
            full_text = ""
            images_found = 0
            figures_found = 0
            ocr_applied = False
            page_count = len(pdf_document)
            
            metadata = {
                "title": pdf_document.metadata.get("title", "Unknown"),
                "author": pdf_document.metadata.get("author", "Unknown"),
                "subject": pdf_document.metadata.get("subject", ""),
                "keywords": pdf_document.metadata.get("keywords", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "creation_date": pdf_document.metadata.get("creationDate", ""),
                "modification_date": pdf_document.metadata.get("modDate", ""),
                "encrypted": pdf_document.needs_pass,
                "file_size_kb": file_size / 1024,
                "total_pages": page_count,
                "images_count": 0,
                "figures_count": 0,
                "ocr_applied": False
            }
            
            for page_num in range(page_count):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if not page_text.strip():
                    image_list = page.get_images()
                    for img_info in image_list:
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        if base_image:
                            image_data = base_image["image"]
                            ocr_text = self.extract_text_from_image(image_data)
                            if ocr_text:
                                page_text += f"\n{ocr_text}"
                                ocr_applied = True
                
                full_text += page_text
                images_found += len(page.get_images())
                figures_found += len(re.findall(r'Fig(?:ure)?\.?\s*\d+', page_text))
            
            metadata.update({
                "images_count": images_found,
                "figures_count": max(images_found, figures_found),
                "ocr_applied": ocr_applied,
                "has_images": images_found > 0 or figures_found > 0
            })
            
            return {
                "text": full_text,
                "metadata": metadata
            }
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}
