# pdf_processor.py

import io
import re
import os
import tempfile
import multiprocessing
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import camelot
import pandas as pd


class PDFProcessor:
    """
    Enhanced PDF file processing and text extraction with:
    - Layout-aware text extraction
    - Table detection and extraction with improved caption finding
    - Visual figure detection with enhanced caption extraction
    - Extended metadata extraction
    - Parallel OCR processing with confidence tracking
    - Section detection (Abstract, Introduction, Discussion, etc.)
    """
    
    def __init__(self, enable_table_extraction: bool = True, enable_visual_figures: bool = True):
        """
        Initialize PDF processor with configuration options.
        
        Args:
            enable_table_extraction: Enable table detection (requires camelot)
            enable_visual_figures: Enable visual figure detection (slower but more accurate)
        """
        self.supported_languages = ['eng']
        
        # Configuration flags
        self.enable_table_extraction = enable_table_extraction
        self.enable_visual_figures = enable_visual_figures
        
        # Auto-detect optimal thread count for parallel OCR
        cpu_count = multiprocessing.cpu_count()
        self.ocr_workers = min(cpu_count, 4)  # Cap at 4 to avoid memory issues
        
        # Image size threshold for figure detection (pixels²)
        self.MIN_FIGURE_SIZE = 10000  # Filters out small icons/logos
        
        # Constants for caption detection
        self.NEAR_SEARCH_MARGIN = 150  # Pixels to search near figure/table (increased from 50)
        self.EXTENDED_SEARCH_MARGIN = 200  # Extended search for difficult cases
        self.FORMAT_TOLERANCE = 5  # Point tolerance for page format detection
        self.MAX_CAPTION_LENGTH = 1000  # Maximum caption length in characters
        
        # Comprehensive figure caption patterns
        self.figure_patterns = [
            # Standard formats with colon
            r'Figure\s+(\d+)\s*:\s*(.{1,1000}?)(?=\n\s*\n|Figure\s+\d+|Table\s+\d+|\Z)',
            r'Fig\.\s*(\d+)\s*:\s*(.{1,1000}?)(?=\n\s*\n|Fig\.\s*\d+|Table\s+\d+|\Z)',
            r'Fig\s+(\d+)\s*:\s*(.{1,1000}?)(?=\n\s*\n|Fig\s+\d+|Table\s+\d+|\Z)',
            
            # With period separator
            r'Figure\s+(\d+)\.\s+(.{1,1000}?)(?=\n\s*\n|Figure\s+\d+|Table\s+\d+|\Z)',
            r'Fig\.\s*(\d+)\.\s+(.{1,1000}?)(?=\n\s*\n|Fig\.\s*\d+|Table\s+\d+|\Z)',
            
            # With dash separator
            r'Figure\s+(\d+)\s*[-–—]\s*(.{1,1000}?)(?=\n\s*\n|Figure\s+\d+|Table\s+\d+|\Z)',
            r'Fig\.\s*(\d+)\s*[-–—]\s*(.{1,1000}?)(?=\n\s*\n|Fig\.\s*\d+|Table\s+\d+|\Z)',
            
            # With sub-figures (a, b, c)
            r'Figure\s+(\d+[a-z])\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|Figure\s+\d+|Table\s+\d+|\Z)',
            r'Fig\.\s*(\d+[a-z])\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|Fig\.\s*\d+|Table\s+\d+|\Z)',
            
            # With parentheses sub-figures
            r'Figure\s+(\d+)\s*\([a-z]\)\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|Figure\s+\d+|Table\s+\d+|\Z)',
            r'Fig\.\s*(\d+)\s*\([a-z]\)\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|Fig\.\s*\d+|Table\s+\d+|\Z)',
            
            # All caps
            r'FIGURE\s+(\d+)\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|FIGURE\s+\d+|TABLE\s+\d+|\Z)',
            r'FIG\.\s*(\d+)\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|FIG\.\s*\d+|TABLE\s+\d+|\Z)',
            
            # Without separator (just space)
            r'Figure\s+(\d+)\s+([A-Z].{1,1000}?)(?=\n\s*\n|Figure\s+\d+|Table\s+\d+|\Z)',
            r'Fig\.\s*(\d+)\s+([A-Z].{1,1000}?)(?=\n\s*\n|Fig\.\s*\d+|Table\s+\d+|\Z)',
        ]
        
        # Comprehensive table caption patterns
# Comprehensive table caption patterns - IMPROVED for academic papers
        self.table_patterns = [
          #  PRIORITY: Nearby patterns (for above-table search)
            # Pattern 1: Colon separator with controlled capture
            r'Table\s+(\d+)\s*:\s*(.+?)(?=\n\s*\n|\n[A-Z][a-z]+\s+\d+|\Z)',
            
            # Pattern 2: Period separator
            r'Table\s+(\d+)\.\s+(.+?)(?=\n\s*\n|\n[A-Z][a-z]+\s+\d+|\Z)',
            
            # Pattern 3: Space-only with STRICT boundary (stops at double newline or new table/fig)
            r'Table\s+(\d+)\s+([A-Z].+?)(?=\n\s*\n|Table\s+\d+|Figure\s+\d+|\Z)',
            
            # Pattern 4: All caps
            r'TABLE\s+(\d+)\s*[:\.]?\s*(.+?)(?=\n\s*\n|TABLE\s+\d+|\Z)',
            
#             #pattern5:Add  for separate-line format: 
            r'Table\s+(\d+)\s*\n+(.+?)(?=\n\s*\n|\Z)'

            # WITH colon/period (existing patterns - keep these)
            r'Table\s+(\d+)\s*:\s*(.{1,1000}?)(?=\n\s*\n|Table\s+\d+|Figure\s+\d+|\Z)',
            r'Table\s+(\d+)\.\s+(.{1,1000}?)(?=\n\s*\n|Table\s+\d+|Figure\s+\d+|\Z)',
            r'Table\s+(\d+)\s*[-–—]\s*(.{1,1000}?)(?=\n\s*\n|Table\s+\d+|Figure\s+\d+|\Z)',
            
            # WITHOUT separator (SPACE ONLY) - NEW for academic papers
            r'Table\s+(\d+)\s+([A-Z][^.!?]*(?:[.!?](?!\s*$)[^.!?]*){0,3}[.!?])',  # Captures up to 3 sentences
            r'TABLE\s+(\d+)\s+([A-Z][^.!?]*(?:[.!?](?!\s*$)[^.!?]*){0,3}[.!?])',
            
            # All caps with various separators
            r'TABLE\s+(\d+)\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|TABLE\s+\d+|FIGURE\s+\d+|\Z)',
            
            # Short forms
            r'Tab\.\s*(\d+)\s*[:\.\-–—]?\s*(.{1,1000}?)(?=\n\s*\n|Tab\.\s*\d+|Figure\s+\d+|\Z)',
        ]
        
        # Section patterns for detection
        self.section_patterns = {
            'abstract': [
                r'\bABSTRACT\b',
                r'\bAbstract\b',
                r'\bSUMMARY\b',
                r'\bSummary\b'
            ],
            'introduction': [
                r'\b(?:1\.?\s*)?INTRODUCTION\b',
                r'\b(?:1\.?\s*)?Introduction\b',
                r'\b(?:I\.?\s*)?INTRODUCTION\b'
            ],
            'methodology': [
                r'\b(?:\d+\.?\s*)?(?:METHODS?|METHODOLOGY)\b',
                r'\b(?:\d+\.?\s*)?(?:Methods?|Methodology)\b',
                r'\b(?:II+\.?\s*)?(?:METHODS?|METHODOLOGY)\b'
            ],
            'results': [
                r'\b(?:\d+\.?\s*)?RESULTS?\b',
                r'\b(?:\d+\.?\s*)?Results?\b',
                r'\b(?:I{2,}V?\.?\s*)?RESULTS?\b'
            ],
            'discussion': [
                r'\b(?:\d+\.?\s*)?DISCUSSION\b',
                r'\b(?:\d+\.?\s*)?Discussion\b',
                r'\b(?:I{2,}V?\.?\s*)?DISCUSSION\b'
            ],
            'conclusion': [
                r'\b(?:\d+\.?\s*)?CONCLUSIONS?\b',
                r'\b(?:\d+\.?\s*)?Conclusions?\b',
                r'\b(?:V+I*\.?\s*)?CONCLUSIONS?\b'
            ],
            'references': [
                r'\bREFERENCES\b',
                r'\bReferences\b',
                r'\bBIBLIOGRAPHY\b',
                r'\bBibliography\b'
            ]
        }
    def calculate_caption_quality(self, caption: str) -> float:
        """
        Score caption quality (0-1, higher = better).
        
        Args:
            caption: Caption text to evaluate
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Length score (optimal range: 20-200 characters)
        length = len(caption)
        if 20 <= length <= 200:
            score += 0.3
        elif 10 <= length <= 300:
            score += 0.15
        elif length > 5:
            score += 0.05
        
        # Has separator (colon/period) = more likely real caption
        if ':' in caption:
            score += 0.25
        elif '. ' in caption and not caption.endswith('.'):
            score += 0.15
        
        # Has descriptive/academic words
        descriptive_words = [
            'shows', 'presents', 'illustrates', 'depicts', 'demonstrates',
            'comparing', 'comparison', 'results', 'analysis', 'distribution',
            'overview', 'summary', 'relationship', 'between', 'effect',
            'performance', 'evaluation', 'measurement', 'data', 'values'
        ]
        words_found = sum(1 for word in descriptive_words if word in caption.lower())
        score += min(words_found * 0.1, 0.25)
        
        # Not just "Figure X" or "Table X" (has substantial text)
        word_count = len(caption.split())
        if word_count > 5:
            score += 0.15
        elif word_count > 3:
            score += 0.1
        elif word_count > 2:
            score += 0.05
        
        return min(score, 1.0)

    def is_likely_reference(self, match_text: str, surrounding_text: str) -> bool:
        """
        Detect if matched text is a reference to a figure/table vs actual caption.
        
        Args:
            match_text: The matched "Figure X" or "Table X" text
            surrounding_text: Context around the match
        
        Returns:
            True if it's likely a reference (not a caption)
        """
        # Extract just the label (e.g., "Figure 3" from "Figure 3: Caption...")
        label_match = re.search(r'((?:Figure|Fig\.|Table|Tab\.)\s*\d+[a-z]?)', match_text, re.IGNORECASE)
        if not label_match:
            return False
        
        label = label_match.group(1)
        
        # Reference indicators (appears mid-sentence)
        reference_patterns = [
            # Preceded by referencing words
            r'(?:as|see|in|from|shown in|described in|presented in|illustrated in|depicted in|given in|listed in)\s+' + re.escape(label),
            
            # In parentheses
            r'\(' + re.escape(label) + r'\)',
            r'\[' + re.escape(label) + r'\]',
            
            # Following verbs (subject reference)
            re.escape(label) + r'\s+(?:shows?|indicates?|demonstrates?|presents?|illustrates?|depicts?|reveals?|suggests?)',
            
            # Possessive or descriptive
            r'(?:the|a|an|this|that|these|those)\s+' + re.escape(label),
            
            # List/series references
            r'(?:Figures?|Tables?)\s+\d+\s*(?:and|,|&)\s*\d+',
            
            # Range references
            r'(?:Figures?|Tables?)\s+\d+\s*[-–]\s*\d+',
        ]
        
        for pattern in reference_patterns:
            if re.search(pattern, surrounding_text, re.IGNORECASE):
                return True  # It's a reference, not a caption
        
        # Caption indicators (starts line, has colon/period after number)
        caption_patterns = [
            # Starts line with label followed by colon/period
            r'(?:^|\n)\s*' + re.escape(label) + r'\s*[:.\-–]',
            
            # Has bold/formatting markers (common in PDFs)
            r'(?:\*\*|__)\s*' + re.escape(label) + r'\s*(?:\*\*|__)\s*[:.\-–]',
            
            # Standalone on line
            r'(?:^|\n)\s*' + re.escape(label) + r'\s*$',
        ]
        
        for pattern in caption_patterns:
            if re.search(pattern, surrounding_text, re.MULTILINE | re.IGNORECASE):
                return False  # It's a caption
        
        # Additional heuristic: if match is at start of surrounding text, likely caption
        if surrounding_text.strip().startswith(label):
            return False
        
        # Default: if in doubt, treat as reference (safer to exclude)
        return True

    def deduplicate_by_number(self, items: List[Dict]) -> List[Dict]:
            """
            Remove duplicates based on BOTH number AND position overlap.
            """
            if not items:
                return []
            
            def boxes_overlap(box1, box2, threshold=0.8):
                """Check if two bounding boxes overlap significantly"""
                if not box1 or not box2:
                    return False
                
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2
                
                # Calculate intersection
                x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                intersection = x_overlap * y_overlap
                
                # Calculate union
                area1 = (x1_max - x1_min) * (y1_max - y1_min)
                area2 = (x2_max - x2_min) * (y2_max - y2_min)
                union = area1 + area2 - intersection
                
                # IoU (Intersection over Union)
                iou = intersection / union if union > 0 else 0
                return iou > threshold
            
            # Group by page first (can't be same table if different pages)
            by_page = {}
            for item in items:
                page = item.get('page', 0)
                if page not in by_page:
                    by_page[page] = []
                by_page[page].append(item)
            
            deduplicated = []
            
            for page, page_items in by_page.items():
                kept = []
                
                for item in page_items:
                    # Check if this item overlaps with any already kept item
                    is_duplicate = False
                    
                    for kept_item in kept:
                        if boxes_overlap(item.get('position'), kept_item.get('position')):
                            # It's a duplicate - keep the one with better caption
                            quality_new = self.calculate_caption_quality(item.get('caption', ''))
                            quality_kept = self.calculate_caption_quality(kept_item.get('caption', ''))
                            
                            if quality_new > quality_kept:
                                # Replace kept item with this better one
                                kept.remove(kept_item)
                                kept.append(item)
                            
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        kept.append(item)
                
                deduplicated.extend(kept)
            
            return deduplicated
    def filter_incomplete_captions(self, items: List[Dict], min_caption_length: int = 10) -> List[Dict]:
        """
        Filter out or mark incomplete captions.
        
        Args:
            items: List of figure/table dictionaries
            min_caption_length: Minimum caption length to consider complete
        
        Returns:
            Filtered list
        """
        filtered = []
        
        for item in items:
            caption = item.get('caption', '')
            
            # Check if caption is just the label (e.g., "Figure 3")
            label_only_pattern = r'^(?:Figure|Fig\.|Table|Tab\.)\s*\d+[a-z]?\s*$'
            is_label_only = re.match(label_only_pattern, caption, re.IGNORECASE)
            
            # Check caption length
            if len(caption) < min_caption_length or is_label_only:
                # Mark as incomplete but keep the entry
                num = item.get('number', '')
                item_type = 'Figure' if 'figure' in str(item).lower() else 'Table'
                item['caption'] = f"{item_type} {num} (caption not found)"
                item['incomplete'] = True
            else:
                item['incomplete'] = False
            
            filtered.append(item)
        
        return filtered

    def validate_and_clean_items(self, items: List[Dict], item_type: str = 'Figure') -> List[Dict]:
        """
        Complete validation and cleaning pipeline.
        
        Args:
            items: List of figure/table dictionaries
            item_type: 'Figure' or 'Table'
        
        Returns:
            Cleaned and validated list
        """
        if not items:
            return []
        
        # Step 1: Filter incomplete captions
        items = self.filter_incomplete_captions(items)
        
        # Step 2: Deduplicate by number (keep best)
        items = self.deduplicate_by_number(items)
        
        # Step 3: Sort by page number and then by figure/table number
        def extract_number(item):
            num_str = str(item.get('number', '0'))
            # Extract numeric part
            match = re.search(r'(\d+)', num_str)
            return int(match.group(1)) if match else 0
        
        items.sort(key=lambda x: (x.get('page', 0), extract_number(x)))
        
        # Step 4: Final validation - ensure no exact duplicates remain
        seen_numbers = set()
        final_items = []
        
        for item in items:
            num = str(item.get('number', ''))
            page = item.get('page', 0)
            key = f"{num}_{page}"
            
            if key not in seen_numbers:
                seen_numbers.add(key)
                final_items.append(item)
        
        return final_items

    def extract_text_from_image(self, image_data: bytes, lang: str = 'eng') -> Dict[str, Any]:
        """
        Extract text from image using OCR with preprocessing and confidence tracking.
        
        Args:
            image_data: Image data as bytes
            lang: Language for OCR (default 'eng')
        
        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image for better OCR accuracy
            image = self._preprocess_image_for_ocr(image)
            
            # Perform OCR with detailed data (includes confidence)
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=lang, 
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            
            for i, conf in enumerate(ocr_data['conf']):
                if int(conf) > 0:  # Valid confidence score
                    text = ocr_data['text'][i]
                    if text.strip():  # Non-empty text
                        texts.append(text)
                        confidences.append(int(conf))
            
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence / 100.0  # Convert to 0-1 scale
            }
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return {'text': '', 'confidence': 0.0}
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Improvements:
        - Convert to grayscale
        - Enhance contrast
        - Resize if too small
        
        Args:
            image: PIL Image object
        
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize if image is too small (OCR works better on larger images)
            min_dimension = 1000
            if min(image.size) < min_dimension:
                scale = min_dimension / min(image.size)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast using PIL's autocontrast
            from PIL import ImageOps
            image = ImageOps.autocontrast(image)
            
            return image
        except Exception as e:
            print(f"Image preprocessing error: {str(e)}")
            return image  # Return original if preprocessing fails
    
    def extract_text_with_layout(self, page) -> str:
        """
        Extract text while preserving reading order and layout.
        Handles multi-column layouts common in research papers.
        
        Args:
            page: PyMuPDF page object
        
        Returns:
            Text string with preserved layout
        """
        try:
            # Get text blocks with position information
            blocks = page.get_text("dict")["blocks"]
            
            # Filter text blocks (type 0 = text, type 1 = image)
            text_blocks = [b for b in blocks if b.get("type") == 0]
            
            if not text_blocks:
                return ""
            
            # Sort blocks by position (handles multi-column layouts)
            sorted_blocks = self._sort_blocks_by_reading_order(text_blocks, page)
            
            # Extract text from sorted blocks
            page_text = ""
            for block in sorted_blocks:
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    page_text += line_text + "\n"
                page_text += "\n"  # Paragraph break between blocks
            
            return page_text
        except Exception as e:
            print(f"Layout extraction error: {str(e)}")
            # Fallback to simple text extraction
            return page.get_text()
    
    def _sort_blocks_by_reading_order(self, blocks: List[Dict], page) -> List[Dict]:
        """
        Sort text blocks to maintain proper reading order.
        Detects columns and sorts accordingly.
        
        Args:
            blocks: List of text block dictionaries
            page: PyMuPDF page object
        
        Returns:
            Sorted list of blocks
        """
        if not blocks:
            return []
        
        try:
            # Get page width
            page_rect = page.rect
            page_width = page_rect.width
            
            # Detect if two-column layout (common in academic papers)
            middle_x = page_width / 2
            
            left_column = [b for b in blocks if b["bbox"][0] < middle_x]
            right_column = [b for b in blocks if b["bbox"][0] >= middle_x]
            
            # If significant content in both columns, treat as multi-column
            if len(left_column) > 2 and len(right_column) > 2:
                # Sort each column by Y position (top to bottom)
                left_sorted = sorted(left_column, key=lambda b: b["bbox"][1])
                right_sorted = sorted(right_column, key=lambda b: b["bbox"][1])
                
                # Return left column first, then right column
                return left_sorted + right_sorted
            else:
                # Single column: sort all by Y position (top to bottom)
                return sorted(blocks, key=lambda b: b["bbox"][1])
        except Exception as e:
            print(f"Block sorting error: {str(e)}")
            # Fallback to simple Y-position sorting
            return sorted(blocks, key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
    
    def _clean_caption_text(self, text: str) -> str:
        """
        Clean and normalize caption text.
        
        Args:
            text: Raw caption text
        
        Returns:
            Cleaned caption text
        """
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove page numbers that might have been captured
        text = re.sub(r'\b\d+\s*$', '', text)
        
        # Trim to reasonable length
        if len(text) > self.MAX_CAPTION_LENGTH:
            text = text[:self.MAX_CAPTION_LENGTH] + "..."
        
        return text.strip()
    
    def _clean_table_caption(self, caption: str) -> str:
        """
        Clean table caption and remove accidentally captured table content.
        
        Args:
            caption: Raw table caption text
        
        Returns:
            Cleaned caption text without table data
        """
        # Remove anything that looks like table data (multiple numbers in sequence)
        caption = re.sub(r'(?:\d+\.?\d*\s+){3,}.*$', '', caption)
        
        # Stop at patterns that indicate table content started
        # e.g., "Model nRMSE RS FS" are column headers
        caption = re.sub(r'\b(?:Model|Method|Parameter|Value|Results?|Training|Testing)\s+[A-Z].*$', '', caption, flags=re.IGNORECASE)
        
        # Remove common table artifacts
        caption = re.sub(r'\b(?:min|max|avg|std|mean)\)\s*\d+.*$', '', caption, flags=re.IGNORECASE)
        
        # Remove trailing incomplete sentences
        sentences = re.split(r'[.!?]', caption)
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            caption = '.'.join(sentences[:-1]) + '.'
        
        # Standard cleaning
        caption = ' '.join(caption.split())
        
        return caption.strip()

    def _validate_caption_number(self, found_number: str, expected_number: int) -> Dict[str, Any]:
        """
        Validate that caption number matches expected table number.
        
        Args:
            found_number: Number found in caption (e.g., "3" from "Table 3")
            expected_number: Expected table number based on detection order
        
        Returns:
            Dictionary with 'valid' and 'confidence_bonus' keys
        """
        try:
            found_num = int(re.search(r'\d+', found_number).group())
            
            # Perfect match
            if found_num == expected_number:
                return {'valid': True, 'confidence_bonus': 0.3}
            
            # Close match (off by 1-2, might be numbering style difference)
            elif abs(found_num - expected_number) <= 2:
                return {'valid': True, 'confidence_bonus': 0.1}
            
            # Significant mismatch - likely wrong caption or reference
            else:
                return {'valid': False, 'confidence_bonus': -0.3}
                
        except (ValueError, AttributeError):
            return {'valid': False, 'confidence_bonus': 0.0}
    
    def _search_text_in_rect(self, page, rect: fitz.Rect) -> str:
        """
        Extract text from a specific rectangle on the page.
        
        Args:
            page: PyMuPDF page object
            rect: Rectangle to search
        
        Returns:
            Text found in rectangle
        """
        try:
            return page.get_text("text", clip=rect)
        except Exception as e:
            print(f"Error extracting text from rect: {str(e)}")
            return ""
    
    def _find_figure_caption_nearby(self, page, img_rect, figure_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for figure caption in the area near the image.
        
        Args:
            page: PyMuPDF page object
            img_rect: Image rectangle (x0, y0, x1, y1)
            figure_num: Expected figure number
        
        Returns:
            Dictionary with 'number', 'caption', 'confidence', and 'method' keys
        """
        if not img_rect:
            return {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'none'}
        
        try:
            x0, y0, x1, y1 = img_rect
            page_height = page.rect.height
            page_width = page.rect.width
            
            # Define search areas (below, above, left, right)
            search_areas = [
                # Below image (most common)
                ('below', fitz.Rect(
                    max(0, x0 - 20),
                    y1,
                    min(page_width, x1 + 20),
                    min(page_height, y1 + self.NEAR_SEARCH_MARGIN)
                )),
                # Above image (some journals)
                ('above', fitz.Rect(
                    max(0, x0 - 20),
                    max(0, y0 - self.NEAR_SEARCH_MARGIN),
                    min(page_width, x1 + 20),
                    y0
                )),
                # Extended below (if not found nearby)
                ('extended_below', fitz.Rect(
                    max(0, x0 - 50),
                    y1,
                    min(page_width, x1 + 50),
                    min(page_height, y1 + self.EXTENDED_SEARCH_MARGIN)
                )),
            ]
            
            best_match = {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'none'}
            
            # Try each search area
            for area_name, search_rect in search_areas:
                caption_text = self._search_text_in_rect(page, search_rect)
                
                if not caption_text.strip():
                    continue
                
                # Try all figure patterns
                for pattern in self.figure_patterns:
                    matches = re.finditer(pattern, caption_text, re.IGNORECASE | re.DOTALL)
                    
                    for match in matches:
                        try:
                            fig_number = match.group(1).strip()
                            fig_caption_text = match.group(2).strip() if len(match.groups()) > 1 else ""
                            
                            # Build full caption
                            full_caption = f"Figure {fig_number}"
                            if fig_caption_text:
                                # Determine separator from original match
                                original_match = match.group(0)
                                if ':' in original_match:
                                    full_caption += f": {fig_caption_text}"
                                elif original_match.count('.') > 1:  # Has period separator (not just "Fig.")
                                    full_caption += f". {fig_caption_text}"
                                else:
                                    full_caption += f" {fig_caption_text}"
                            
                            full_caption = self._clean_caption_text(full_caption)
                            
                            # Calculate confidence score
                            confidence = self._calculate_caption_confidence(
                                fig_number, 
                                full_caption, 
                                figure_num,
                                area_name
                            )
                            
                            # Keep best match
                            if confidence > best_match['confidence']:
                                best_match = {
                                    'number': fig_number,
                                    'caption': full_caption,
                                    'confidence': confidence,
                                    'method': f'nearby_{area_name}'
                                }
                        except (IndexError, AttributeError) as e:
                            continue
            
            return best_match
            
        except Exception as e:
            print(f"Error in nearby caption search: {str(e)}")
            return {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'error'}
    
    def _find_figure_caption_full_page(self, page, figure_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Search full page with ADVANCED FILTERING (removes references).
        
        Args:
            page: PyMuPDF page object
            figure_num: Expected figure number
        
        Returns:
            Dictionary with 'number', 'caption', 'confidence', and 'method' keys
        """
        try:
            # Get full page text
            page_text = page.get_text()
            
            if not page_text.strip():
                return {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'none'}
            
            best_match = {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'none'}
            
            # Try all figure patterns
            for pattern in self.figure_patterns:
                matches = re.finditer(pattern, page_text, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    try:
                        # Get full match and context
                        full_match = match.group(0)
                        
                        # Extract context (200 chars before/after)
                        start = max(0, match.start() - 200)
                        end = min(len(page_text), match.end() + 200)
                        context = page_text[start:end]
                        
                        # FILTER: Check if this is a reference, not a caption
                        if self.is_likely_reference(full_match, context):
                            continue  # Skip references
                        
                        # Extract figure number and caption
                        fig_number = match.group(1).strip()
                        fig_caption_text = match.group(2).strip() if len(match.groups()) > 1 else ""
                        
                        # Build full caption
                        full_caption = f"Figure {fig_number}"
                        if fig_caption_text and len(fig_caption_text) > 3:
                            # Determine separator
                            if ':' in full_match[:20]:
                                full_caption += f": {fig_caption_text}"
                            elif full_match.count('.') > 1:
                                full_caption += f". {fig_caption_text}"
                            else:
                                full_caption += f" {fig_caption_text}"
                        
                        full_caption = self._clean_caption_text(full_caption)
                        
                        # Calculate quality-based confidence
                        quality = self.calculate_caption_quality(full_caption)
                        confidence = quality * 0.7  # Reduce for full-page
                        
                        # Bonus if number matches expected
                        if figure_num is not None:
                            try:
                                found_num = int(re.search(r'\d+', fig_number).group())
                                if found_num == figure_num:
                                    confidence += 0.2
                            except (ValueError, AttributeError):
                                pass
                        
                        # Keep best match
                        if confidence > best_match['confidence']:
                            best_match = {
                                'number': fig_number,
                                'caption': full_caption,
                                'confidence': confidence,
                                'method': 'full_page_filtered'
                            }
                            
                    except (IndexError, AttributeError):
                        continue
            
            return best_match
            
        except Exception as e:
            print(f"Error in full-page figure caption search: {str(e)}")
            return {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'error'}
    
    def _calculate_caption_confidence(
        self, 
        found_number: str, 
        caption: str, 
        expected_number: Optional[int],
        search_method: str
    ) -> float:
        """
        Calculate confidence score for a found caption.
        
        Args:
            found_number: Figure/table number found in caption
            caption: Full caption text
            expected_number: Expected figure/table number based on position
            search_method: How caption was found ('below', 'above', 'full_page', etc.)
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        
        # Base score for finding any caption
        confidence += 0.3
        
        # Number matching (if we know expected number)
        if expected_number is not None:
            try:
                # Extract numeric part
                found_num = int(re.search(r'\d+', found_number).group())
                if found_num == expected_number:
                    confidence += 0.3
                elif abs(found_num - expected_number) <= 1:
                    confidence += 0.15  # Close numbers get some credit
            except (ValueError, AttributeError):
                pass
        
        # Caption length (reasonable length indicates real caption)
        caption_length = len(caption)
        if 20 <= caption_length <= 500:
            confidence += 0.2
        elif caption_length > 10:
            confidence += 0.1
        
        # Search method bonus
        method_bonuses = {
            'below': 0.15,  # Most common position
            'extended_below': 0.1,
            'above': 0.05,
            'full_page': 0.0  # No bonus for full-page search
        }
        confidence += method_bonuses.get(search_method, 0.0)
        
        # Bonus for having descriptive text (not just "Figure X")
        if ':' in caption or '.' in caption:
            words = caption.split()
            if len(words) > 3:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _find_figure_caption(self, page, img_rect, figure_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Find figure caption using hybrid approach: nearby search first, then full-page fallback.
        
        Args:
            page: PyMuPDF page object
            img_rect: Image rectangle
            figure_num: Expected figure number
        
        Returns:
            Dictionary with caption information
        """
        # Method 1: Search near image (fast, high confidence)
        nearby_result = self._find_figure_caption_nearby(page, img_rect, figure_num)
        
        if nearby_result['confidence'] > 0.6:
            return nearby_result
        
        # Method 2: Search full page (slower, lower confidence)
        full_page_result = self._find_figure_caption_full_page(page, figure_num)
        
        # Return best result
        if full_page_result['confidence'] > nearby_result['confidence']:
            return full_page_result
        elif nearby_result['confidence'] > 0:
            return nearby_result
        else:
            return {
                'number': str(figure_num) if figure_num else None,
                'caption': f"Figure {figure_num}" if figure_num else "Figure",
                'confidence': 0.0,
                'method': 'fallback'
            }
    
    def _find_table_caption_nearby(self, page, table_bbox, table_num: int) -> Dict[str, Any]:
        """
        Search for table caption near the table (above, below, or inside).
        
        Args:
            page: PyMuPDF page object
            table_bbox: Table bounding box
            table_num: Table number
        
        Returns:
            Dictionary with 'number', 'caption', 'confidence', and 'method' keys
        """
        try:
            x0, y0, x1, y1 = table_bbox
            page_height = page.rect.height
            page_width = page.rect.width
            
            # Define search areas
            search_areas = [
                # Above table (common in many journals)
                ('above', fitz.Rect(
                    max(0, x0 - 20),
                    max(0, y0 - self.NEAR_SEARCH_MARGIN),
                    min(page_width, x1 + 20),
                    y0
                )),
                # Below table (also common)
                ('below', fitz.Rect(
                    max(0, x0 - 20),
                    y1,
                    min(page_width, x1 + 20),
                    min(page_height, y1 + self.NEAR_SEARCH_MARGIN)
                )),
                # Extended above
                ('extended_above', fitz.Rect(
                    max(0, x0 - 50),
                    max(0, y0 - self.EXTENDED_SEARCH_MARGIN),
                    min(page_width, x1 + 50),
                    y0
                )),
            ]
            
            best_match = {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'none'}
            
            # Try each search area
            for area_name, search_rect in search_areas:
                caption_text = self._search_text_in_rect(page, search_rect)
                
                if not caption_text.strip():
                    continue
                
                # Try all table patterns
                for pattern in self.table_patterns:
                    matches = re.finditer(pattern, caption_text, re.IGNORECASE | re.DOTALL)
                    
                    for match in matches:
                        try:
                            tbl_number = match.group(1).strip()
                            tbl_caption_text = match.group(2).strip() if len(match.groups()) > 1 else ""
                            
                            # Build full caption
                            full_caption = f"Table {tbl_number}"
                            if tbl_caption_text:
                                original_match = match.group(0)
                                if ':' in original_match:
                                    full_caption += f": {tbl_caption_text}"
                                elif original_match.count('.') > 1:
                                    full_caption += f". {tbl_caption_text}"
                                else:
                                    full_caption += f" {tbl_caption_text}"
                            
                            # ✅ STEP 2A: Clean the caption (remove table data)
                            full_caption = self._clean_table_caption(full_caption)
                            full_caption = self._clean_caption_text(full_caption)  # existing cleaning
                            
                            # ✅ STEP 2B: Validate caption number
                            validation = self._validate_caption_number(tbl_number, table_num)

                            
                            # Calculate confidence
                            confidence = self._calculate_caption_confidence(
                                tbl_number,
                                full_caption,
                                table_num,
                                area_name
                            )
                            
# ✅ STEP 2C: Apply validation bonus/penalty
                            confidence += validation['confidence_bonus']
                            confidence = max(0.0, min(confidence, 1.0))  # Clamp to [0, 1]

                            # Bonus for finding above (more common for tables)
                            if 'above' in area_name:
                                confidence += 0.1
                            
                            # Keep best match
                            if confidence > best_match['confidence']:
                                best_match = {
                                    'number': tbl_number,
                                    'caption': full_caption,
                                    'confidence': confidence,
                                    'method': f'nearby_{area_name}'
                                }
                        except (IndexError, AttributeError):
                            continue
            
            return best_match
            
        except Exception as e:
            print(f"Error in nearby table caption search: {str(e)}")
            return {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'error'}
    
    def _find_table_caption_full_page(self, page, table_num: int) -> Dict[str, Any]:
        """
        Search full page with ADVANCED FILTERING (removes references).
        """
        try:
            page_text = page.get_text()
            
            if not page_text.strip():
                return {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'none'}
            
            best_match = {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'none'}
            
            # Try all table patterns
            for pattern in self.table_patterns:
                matches = re.finditer(pattern, page_text, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    try:
                        full_match = match.group(0)
                        
                        # Extract context
                        start = max(0, match.start() - 200)
                        end = min(len(page_text), match.end() + 200)
                        context = page_text[start:end]
                        
                        # FILTER: Check if this is a reference
                        if self.is_likely_reference(full_match, context):
                            continue
                        
                        # Extract table number and caption
                        tbl_number = match.group(1).strip()
                        tbl_caption_text = match.group(2).strip() if len(match.groups()) > 1 else ""
                        
                        # Build full caption
                        full_caption = f"Table {tbl_number}"
                        if tbl_caption_text and len(tbl_caption_text) > 3:
                            if ':' in full_match[:20]:
                                full_caption += f": {tbl_caption_text}"
                            elif full_match.count('.') > 1:
                                full_caption += f". {tbl_caption_text}"
                            else:
                                full_caption += f" {tbl_caption_text}"
                        
                        # ✅ STEP 3A: Clean the caption
                        full_caption = self._clean_table_caption(full_caption)
                        full_caption = self._clean_caption_text(full_caption)
                        
                        # ✅ STEP 3B: Validate caption number
                        validation = self._validate_caption_number(tbl_number, table_num)
                        
                        # Calculate quality-based confidence
                        quality = self.calculate_caption_quality(full_caption)
                        confidence = quality * 0.7  # Reduce for full-page search
                        
                        # ✅ STEP 3C: Apply validation bonus/penalty
                        confidence += validation['confidence_bonus']
                        confidence = max(0.0, min(confidence, 1.0))
                        
                        # Keep best match
                        if confidence > best_match['confidence']:
                            best_match = {
                                'number': tbl_number,
                                'caption': full_caption,
                                'confidence': confidence,
                                'method': 'full_page_filtered',
                                'validated': validation['valid']
                            }
                            
                    except (IndexError, AttributeError):
                        continue
            
            return best_match
            
        except Exception as e:
            print(f"Error in full-page table caption search: {str(e)}")
            return {'number': None, 'caption': None, 'confidence': 0.0, 'method': 'error'}
    
    # def _find_table_caption(self, page, table_bbox, table_num: int) -> str:
    #     """
    #     Find table caption using FULL PAGE SEARCH FIRST (better for academic papers).
        
    #     Args:
    #         page: PyMuPDF page object
    #         table_bbox: Table bounding box
    #         table_num: Table number
        
    #     Returns:
    #         Caption string
    #     """
    #     # METHOD 1: Search ENTIRE page first (academic papers have captions far from tables)
    #     full_page_result = self._find_table_caption_full_page(page, table_num)
        
    #     if full_page_result['confidence'] > 0.5:
    #         return full_page_result['caption'] or f"Table {table_num}"
        
    #     # METHOD 2: Fallback to nearby search (for scanned PDFs or when full page fails)
    #     nearby_result = self._find_table_caption_nearby(page, table_bbox, table_num)
        
    #     # Return best result
    #     if nearby_result['confidence'] > full_page_result['confidence']:
    #         return nearby_result['caption'] or f"Table {table_num}"
    #     elif full_page_result['caption']:
    #         return full_page_result['caption']
    #     else:
    #         return f"Table {table_num}"
    def _find_table_caption(self, page, table_bbox, table_num: int) -> str:
        """
        Find table caption - PRIORITIZE nearby above search (most common for tables).
        """
        # METHOD 1: Search ABOVE table first (most reliable for academic papers)
        nearby_result = self._find_table_caption_nearby(page, table_bbox, table_num)
        
        if nearby_result['confidence'] > 0.5:  # Good enough confidence
            return nearby_result['caption'] or f"Table {table_num}"
        
        # METHOD 2: Try full page only if nearby failed completely
        if nearby_result['confidence'] < 0.3:  # Very low confidence
            full_page_result = self._find_table_caption_full_page(page, table_num)
            
            if full_page_result['confidence'] > nearby_result['confidence']:
                return full_page_result['caption'] or f"Table {table_num}"
        
        # Return best available
        return nearby_result['caption'] or f"Table {table_num}"
    
    def extract_tables_pymupdf(self, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables using PyMuPDF's built-in table detection.
        This is a fallback method when Camelot is not available.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
        
        Returns:
            List of table dictionaries
        """
        tables = []
        
        try:
            # Find table structures using PyMuPDF
            tabs = page.find_tables()
            
            for idx, tab in enumerate(tabs):
                # Extract table as pandas DataFrame
                df = tab.to_pandas()
                
                # Find table caption
                table_bbox = tab.bbox
                caption = self._find_table_caption(page, table_bbox, idx + 1)
                
                tables.append({
                    'number': idx + 1,
                    'caption': caption,
                    'data': df,
                    'position': tab.bbox,
                    'text_representation': df.to_string(index=False),
                    'row_count': len(df),
                    'col_count': len(df.columns),
                    'page': page_num + 1
                })
        except Exception as e:
            print(f"PyMuPDF table extraction error on page {page_num}: {str(e)}")
        
        return tables
    
    def detect_and_extract_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF page using Camelot.
        More accurate than PyMuPDF for complex tables.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Zero-indexed page number
        
        Returns:
            List of table dictionaries with data and metadata
        """
        try:
            # Camelot uses 1-indexed pages
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_num + 1),
                flavor='lattice',  # 'lattice' for bordered tables
                suppress_stdout=True
            )
            
            # Open PDF to search for captions
            pdf_doc = fitz.open(pdf_path)
            page = pdf_doc[page_num]
            
            extracted_tables = []
            for i, table in enumerate(tables):
                # Find caption
                table_bbox = table._bbox
                caption = self._find_table_caption(page, table_bbox, i + 1)
                
                extracted_tables.append({
                    'number': i + 1,
                    'caption': caption,
                    'data': table.df,
                    'accuracy': table.accuracy,
                    'position': table._bbox,
                    'text_representation': table.df.to_string(index=False),
                    'row_count': len(table.df),
                    'col_count': len(table.df.columns),
                    'whitespace': table.whitespace,
                    'page': page_num + 1
                })
            
            pdf_doc.close()
            
            # If lattice fails, try stream method
            if not extracted_tables:
                tables_stream = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    flavor='stream',
                    suppress_stdout=True
                )
                
                pdf_doc = fitz.open(pdf_path)
                page = pdf_doc[page_num]
                
                for i, table in enumerate(tables_stream):
                    table_bbox = table._bbox
                    caption = self._find_table_caption(page, table_bbox, i + 1)
                    
                    extracted_tables.append({
                        'number': i + 1,
                        'caption': caption,
                        'data': table.df,
                        'accuracy': table.accuracy,
                        'position': table._bbox,
                        'text_representation': table.df.to_string(index=False),
                        'row_count': len(table.df),
                        'col_count': len(table.df.columns),
                        'page': page_num + 1
                    })
                
                pdf_doc.close()
            
            return extracted_tables
        except Exception as e:
            print(f"Camelot table extraction error on page {page_num}: {str(e)}")
            return []
    
    def detect_figures_visual(self, page, pdf_document, page_num: int) -> List[Dict[str, Any]]:
        """
        Detect figures by analyzing images on the page.
        More comprehensive than regex-only approach.
        
        Args:
            page: PyMuPDF page object
            pdf_document: PyMuPDF document object
            page_num: Page number (0-indexed)
        
        Returns:
            List of figure dictionaries with metadata
        """
        figures_info = []
        
        # Get all images on page
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            
            try:
                # Extract image metadata
                base_image = pdf_document.extract_image(xref)
                
                if base_image:
                    img_width = base_image.get("width", 0)
                    img_height = base_image.get("height", 0)
                    img_area = img_width * img_height
                    
                    # Filter out small images (likely icons/logos)
                    if img_area > self.MIN_FIGURE_SIZE:
                        # Get image position on page
                        img_rects = page.get_image_rects(xref)
                        
                        # Find figure caption
                        caption_result = self._find_figure_caption(
                            page,
                            img_rects[0] if img_rects else None,
                            img_index + 1
                        )
                        
                        figure_data = {
                            'number': caption_result['number'] or str(img_index + 1),
                            'caption': caption_result['caption'] or f"Figure {img_index + 1}",
                            'page': page_num + 1,
                            'index': img_index,
                            'width': img_width,
                            'height': img_height,
                            'area': img_area,
                            'position': list(img_rects[0]) if img_rects else None,
                            'format': base_image.get("ext", "unknown"),
                            'colorspace': base_image.get("colorspace", "unknown"),
                            'xref': xref,
                            'confidence': caption_result['confidence']
                        }
                        
                        figures_info.append(figure_data)
            
            except Exception as e:
                print(f"Error analyzing image {img_index} on page {page_num}: {str(e)}")
                continue
        
        return figures_info
    
    def extract_text_from_images_parallel(
        self, 
        image_list: List[Tuple], 
        pdf_document, 
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images concurrently using ThreadPoolExecutor.
        
        Args:
            image_list: List of image info tuples from PyMuPDF
            pdf_document: PDF document object
            max_workers: Number of parallel OCR threads (default: auto-detected)
        
        Returns:
            List of dictionaries with 'index', 'text', and 'confidence' keys
        """
        if max_workers is None:
            max_workers = self.ocr_workers
        
        results = []
        
        def process_single_image(img_index: int, img_info: Tuple) -> Dict[str, Any]:
            """Helper function to process one image"""
            try:
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                
                if base_image:
                    image_data = base_image["image"]
                    ocr_result = self.extract_text_from_image(image_data)
                    return {
                        'index': img_index,
                        'text': ocr_result['text'],
                        'confidence': ocr_result['confidence']
                    }
            except Exception as e:
                print(f"Error processing image {img_index}: {str(e)}")
            
            return {'index': img_index, 'text': '', 'confidence': 0.0}
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_image, idx, img_info): idx 
                for idx, img_info in enumerate(image_list)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    img_idx = future_to_index[future]
                    print(f"Image {img_idx} generated an exception: {e}")
                    results.append({'index': img_idx, 'text': '', 'confidence': 0.0})
        
        # Sort by original index to maintain order
        results.sort(key=lambda x: x['index'])
        
        return results
    
    def detect_sections(self, text: str) -> Dict[str, Any]:
        """
        Detect standard research paper sections in the text.
        
        Args:
            text: Full text of the document
        
        Returns:
            Dictionary with section information
        """
        sections = {}
        
        for section_name, patterns in self.section_patterns.items():
            found = False
            section_text = ""
            section_start = -1
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    found = True
                    section_start = match.start()
                    
                    # Try to extract section text (up to next section or 1000 chars)
                    remaining_text = text[section_start:]
                    
                    # Find next section header (any section)
                    next_section_match = None
                    min_next_pos = len(remaining_text)
                    
                    for other_patterns in self.section_patterns.values():
                        for other_pattern in other_patterns:
                            next_match = re.search(
                                other_pattern, 
                                remaining_text[100:],  # Skip first 100 chars (current section)
                                re.MULTILINE
                            )
                            if next_match and next_match.start() < min_next_pos:
                                min_next_pos = next_match.start() + 100
                                next_section_match = next_match
                    
                    if next_section_match:
                        section_text = remaining_text[:min_next_pos]
                    else:
                        section_text = remaining_text[:1000]  # Max 1000 chars
                    
                    break
            
            sections[section_name] = {
                'found': found,
                'text': section_text.strip() if found else "",
                'word_count': len(section_text.split()) if found else 0,
                'position': section_start if found else -1
            }
        
        return sections
    
    def extract_abstract(self, text: str) -> Dict[str, Any]:
        """
        Extract abstract from paper text.
        
        Args:
            text: Full text of the document
        
        Returns:
            Dictionary with abstract information
        """
        # Try pattern-based extraction first
        abstract_patterns = [
            r'ABSTRACT\s*\n(.*?)(?:\n\s*\n|\nINTRODUCTION|\n1\.)',
            r'Abstract\s*\n(.*?)(?:\n\s*\n|\nIntroduction|\n1\.)',
            r'ABSTRACT\s*[:\-]?\s*(.*?)(?:\n\s*\n|\nINTRODUCTION|\n1\.)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract_text = match.group(1).strip()
                # Clean up
                abstract_text = ' '.join(abstract_text.split())
                
                return {
                    'found': True,
                    'text': abstract_text,
                    'word_count': len(abstract_text.split()),
                    'char_count': len(abstract_text)
                }
        
        return {
            'found': False,
            'text': '',
            'word_count': 0,
            'char_count': 0
        }
    
    def extract_extended_metadata(self, pdf_document) -> Dict[str, Any]:
        """
        Extract additional metadata beyond basic info.
        
        Args:
            pdf_document: PyMuPDF document object
        
        Returns:
            Dictionary with extended metadata
        """
        extended_meta = {}
        
        try:
            # Language detection (from metadata or content)
            lang = pdf_document.metadata.get("language", None)
            if not lang and pdf_document.page_count > 0:
                # Try to detect from first page text
                first_page_text = pdf_document[0].get_text()[:1000]
                lang = self._detect_language(first_page_text)
            extended_meta['language'] = lang or "unknown"
            
            # Page dimensions and orientation
            if pdf_document.page_count > 0:
                first_page = pdf_document[0]
                rect = first_page.rect
                width = rect.width
                height = rect.height
                
                extended_meta['page_width_pts'] = round(width, 2)
                extended_meta['page_height_pts'] = round(height, 2)
                extended_meta['page_width_inches'] = round(width / 72, 2)
                extended_meta['page_height_inches'] = round(height / 72, 2)
                
                # Determine orientation
                extended_meta['orientation'] = 'landscape' if width > height else 'portrait'
                
                # Determine page size format
                extended_meta['page_format'] = self._identify_page_format(width, height)
            
            # Check if PDF is searchable (has text layer)
            extended_meta['is_searchable'] = self._check_if_searchable(pdf_document)
            
            # Count total words (estimate)
            if pdf_document.page_count > 0:
                # Sample first 3 pages for word count estimation
                sample_pages = min(3, pdf_document.page_count)
                total_chars = sum(
                    len(pdf_document[i].get_text()) 
                    for i in range(sample_pages)
                )
                avg_chars_per_page = total_chars / sample_pages
                total_estimated_chars = avg_chars_per_page * pdf_document.page_count
                extended_meta['estimated_word_count'] = int(total_estimated_chars // 5)
            else:
                extended_meta['estimated_word_count'] = 0
            
            # PDF version
            pdf_version = pdf_document.metadata.get('format', 'Unknown')
            extended_meta['pdf_version'] = f"PDF {pdf_version}" if pdf_version != 'Unknown' else 'Unknown'
            
            # Check if PDF/A (archival format)
            extended_meta['is_pdfa'] = 'PDF/A' in str(pdf_version)
            
        except Exception as e:
            print(f"Extended metadata extraction error: {str(e)}")
        
        return extended_meta
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection using character patterns.
        
        Args:
            text: Text sample to analyze
        
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            if not text.strip():
                return "unknown"
            
            # Simple heuristic: English has high frequency of common words
            english_common = ['the', 'of', 'and', 'to', 'a', 'in', 'is', 'that', 'for', 'it']
            text_lower = text.lower()
            
            english_score = sum(1 for word in english_common if f' {word} ' in f' {text_lower} ')
            
            if english_score >= 4:
                return "en"
            
            return "unknown"
        except Exception as e:
            print(f"Language detection error: {str(e)}")
            return "unknown"
    
    def _identify_page_format(self, width_pts: float, height_pts: float) -> str:
        """
        Identify standard page formats (A4, Letter, Legal, etc.)
        
        Args:
            width_pts: Page width in points
            height_pts: Page height in points
        
        Returns:
            Page format name or custom dimensions
        """
        # Common formats in points (1 inch = 72 points)
        formats = {
            'Letter': (612, 792),
            'A4': (595, 842),
            'Legal': (612, 1008),
            'Tabloid': (792, 1224),
            'A3': (842, 1191),
            'A5': (420, 595)
        }
        
        # Check both portrait and landscape orientations
        for format_name, (w, h) in formats.items():
            # Portrait
            if (abs(width_pts - w) < self.FORMAT_TOLERANCE and 
                abs(height_pts - h) < self.FORMAT_TOLERANCE):
                return format_name
            
            # Landscape
            if (abs(width_pts - h) < self.FORMAT_TOLERANCE and 
                abs(height_pts - w) < self.FORMAT_TOLERANCE):
                return f"{format_name} (Landscape)"
        
        return f"Custom ({width_pts:.0f}x{height_pts:.0f} pts)"
    
    def _check_if_searchable(self, pdf_document) -> bool:
        """
        Check if PDF has text layer (searchable) or is image-only (scanned).
        
        Args:
            pdf_document: PyMuPDF document object
        
        Returns:
            True if PDF has searchable text, False otherwise
        """
        try:
            # Sample first few pages
            sample_pages = min(3, pdf_document.page_count)
            
            for i in range(sample_pages):
                text = pdf_document[i].get_text().strip()
                if len(text) > 50:  # Has substantial text
                    return True
            
            return False
        except Exception as e:
            print(f"Searchability check error: {str(e)}")
            return False
    
    def extract_text_from_pdf(
        self, 
        pdf_file, 
        password: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Main method: Extract text and metadata from PDF with all enhancements.
        
        Args:
            pdf_file: File object or path to PDF
            password: Optional password for encrypted PDFs
            progress_callback: Optional callback function(current_page, total_pages)
        
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Document metadata with OCR info, sections, figures, tables
                - tables: List of extracted tables (if enabled)
                - figures: List of detected figures (if enabled)
                - sections: Detected paper sections
                - error: Error message (if failed)
        """
        temp_pdf_path = None
        
        try:
            # Read PDF into memory
            pdf_bytes = pdf_file.read()
            file_size = len(pdf_bytes)
            
            # Save to temp file for Camelot (it requires file path)
            if self.enable_table_extraction:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_bytes)
                    temp_pdf_path = tmp_file.name
            
            try:
                # Open PDF with PyMuPDF
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                # Check for password protection
                if pdf_document.needs_pass:
                    if password:
                        if not pdf_document.authenticate(password):
                            return {"error": "Invalid password"}
                    else:
                        return {"error": "PDF is encrypted, password required"}
            
            except Exception as e:
                return {"error": f"Error opening PDF: {str(e)}"}
            
            # Initialize tracking variables
            full_text = ""
            all_tables = []
            all_figures = []
            images_found = 0
            figures_found_by_regex = 0
            page_count = len(pdf_document)
            
            # OCR tracking
            ocr_applied = False
            ocr_pages = []
            ocr_confidences = []
            
            # Extract basic metadata
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
                "file_size_kb": round(file_size / 1024, 2),
                "total_pages": page_count,
            }
            
            # Extract extended metadata
            extended_meta = self.extract_extended_metadata(pdf_document)
            metadata.update(extended_meta)
            
            # Process each page
            for page_num in range(page_count):
                page = pdf_document[page_num]
                
                # Report progress if callback provided
                if progress_callback:
                    progress_callback(page_num + 1, page_count)
                
                # Extract text with layout preservation
                page_text = self.extract_text_with_layout(page)
                
                # Extract tables if enabled
                if self.enable_table_extraction and temp_pdf_path:
                    try:
                        page_tables = self.detect_and_extract_tables(temp_pdf_path, page_num)
                    except Exception as e:
                        print(f"Camelot failed on page {page_num}, falling back to PyMuPDF: {str(e)}")
                        page_tables = self.extract_tables_pymupdf(page, page_num)
                    
                    if page_tables:
                        all_tables.extend(page_tables)
                        # Add table markers to text
                        for table in page_tables:
                            page_text += f"\n\n[TABLE {table['number']} - Page {page_num + 1}]\n"
                            page_text += f"{table['caption']}\n"
                            page_text += table['text_representation']
                            page_text += f"\n[END TABLE {table['number']}]\n\n"
                
                # Detect figures visually if enabled
                if self.enable_visual_figures:
                    detected_figures = self.detect_figures_visual(page, pdf_document, page_num)
                    all_figures.extend(detected_figures)
                
                # OCR processing for pages with no/little text
                if len(page_text.strip()) < 50:  # Likely scanned page
                    image_list = page.get_images()
                    
                    if image_list:
                        # Use parallel OCR processing
                        ocr_results = self.extract_text_from_images_parallel(
                            image_list, 
                            pdf_document,
                            max_workers=self.ocr_workers
                        )
                        
                        # Combine OCR results and track confidence
                        page_ocr_applied = False
                        for ocr_data in ocr_results:
                            if ocr_data['text'].strip():
                                page_text += f"\n[OCR from image {ocr_data['index'] + 1}]\n{ocr_data['text']}\n"
                                ocr_applied = True
                                page_ocr_applied = True
                                ocr_confidences.append(ocr_data['confidence'])
                        
                        if page_ocr_applied:
                            ocr_pages.append(page_num + 1)  # 1-indexed for display
                
                # Add page text to full document text
                full_text += f"\n--- Page {page_num + 1} ---\n"
                full_text += page_text
                
                # Count images and figure references
                images_found += len(page.get_images())
                figures_found_by_regex += len(re.findall(r'Fig(?:ure)?\.?\s*\d+', page_text, re.IGNORECASE))
            
            # Close PDF document
            pdf_document.close()
            
            # Detect sections in the full text
            detected_sections = self.detect_sections(full_text)
            
            # Extract abstract
            abstract_info = self.extract_abstract(full_text)
            
            ####### ADVANCED CLEANING PIPELINE 
            figure_list = []
            if self.enable_visual_figures and all_figures:
                # Apply validation and cleaning
                cleaned_figures = self.validate_and_clean_items(all_figures, 'Figure')
                
                for fig in cleaned_figures:
                    caption = fig['caption']
                    
                    # Skip if caption not found
                    if '(caption not found)' in caption:
                        continue
                    
                    # Skip if it's just "Figure X" with no description
                    if caption.strip() == f"Figure {fig['number']}":
                        continue
                    
                    figure_list.append({
                        'number': fig['number'],
                        'caption': fig['caption'],
                        'page': fig['page']
                    })


            # Clean and deduplicate TABLES
            table_list = []
            if all_tables:
                # Apply validation and cleaning
                cleaned_tables = self.validate_and_clean_items(all_tables, 'Table')
                
                for table in cleaned_tables:
                    caption = table['caption']
                    
                    # Skip if caption not found
                    if '(caption not found)' in caption:
                        continue
                    
                    # Skip if it's just "Table X" with no description
                    if caption.strip() == f"Table {table['number']}":
                        continue
                    
                    table_list.append({
                        'number': table['number'],
                        'caption': table['caption'],
                        'page': table['page']
                    })

            # ============= END CLEANING PIPELINE
            
            # Calculate OCR statistics
            ocr_confidence_avg = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0
            
            # Determine document type based on OCR usage
            if not ocr_applied:
                document_type = "native"  # Born-digital PDF
            elif len(ocr_pages) == page_count:
                document_type = "scanned"  # Fully scanned document
            else:
                document_type = "mixed"  # Mix of native and scanned pages
            
            # Update metadata with final counts and OCR info
            metadata.update({
                # Image/Figure/Table counts
                "images_count": images_found,
                "figures_count": len(all_figures) if self.enable_visual_figures else figures_found_by_regex,
                "tables_count": len(all_tables),
                "has_images": images_found > 0,
                "has_tables": len(all_tables) > 0,
                "has_figures": len(all_figures) > 0 if self.enable_visual_figures else figures_found_by_regex > 0,
                
                # OCR information
                "ocr_applied": ocr_applied,
                "ocr_page_count": len(ocr_pages),
                "ocr_pages": ocr_pages,
                "ocr_confidence_avg": round(ocr_confidence_avg, 3),
                "document_type": document_type,
                
                # Section information
                "has_abstract": abstract_info['found'],
                "abstract_word_count": abstract_info['word_count'],
                "has_introduction": detected_sections['introduction']['found'],
                "has_methodology": detected_sections['methodology']['found'],
                "has_results": detected_sections['results']['found'],
                "has_discussion": detected_sections['discussion']['found'],
                "has_conclusion": detected_sections['conclusion']['found'],
                "has_references": detected_sections['references']['found'],
                
                # Content lists
                "figure_list": figure_list,
                "table_list": table_list
            })
            
            # Prepare return dictionary
            result = {
                "text": full_text,
                "metadata": metadata,
                "abstract": abstract_info,
                "sections": detected_sections
            }
            
            # Add tables if extracted
            if self.enable_table_extraction and all_tables:
                result["tables"] = all_tables
            
            # Add figures if detected
            if self.enable_visual_figures and all_figures:
                result["figures"] = all_figures
            
            return result
            
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}
        
        finally:
            # Clean up temporary file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    print(f"Failed to delete temp file: {str(e)}")