# citation_analyzer_semantic_hybrid_FIXED.py

import re
import time
import logging
import networkx as nx
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict
import numpy as np

# Configure logging to file
def setup_logger(log_file: str = 'citation_analyzer.log', level=logging.INFO):
    """Setup logger to write to file with timestamps."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    
    # Formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

# Initialize logger (will be set up properly when class is instantiated)
logger = logging.getLogger(__name__)

class HybridSemanticCitationAnalyzer:
    """
    FIXED Hybrid citation analyzer combining semantic similarity with co-occurrence.
    
    
    Enhanced Features:
     Semantic similarity using sentence embeddings
     Co-occurrence detection (citations in same sentence/paragraph)
     spaCy for accurate sentence boundary detection
     Year distribution with context validation
     Stance detection (supporting, refuting, neutral, contrasting)
     Purpose classification (background, methodology, comparison, etc.)
     Hybrid network with multiple relationship types
    """

    # Stopwords that should NEVER be extracted as citations
    AUTHOR_STOPWORDS = {
        # Common words at sentence start
        'the', 'this', 'that', 'these', 'those', 'a', 'an',
        'after', 'before', 'during', 'while', 'when', 'where',
        'however', 'moreover', 'furthermore', 'therefore', 'thus',
        'in', 'on', 'at', 'by', 'with', 'from', 'to', 'for',
        
        # Month names
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        
        # Common academic words
        'report', 'study', 'research', 'paper', 'article', 'publication',
        'journal', 'conference', 'proceedings', 'abstract', 'review',
        'analysis', 'model', 'method', 'results', 'conclusion',
        'international', 'national', 'annual', 'quarterly',
        
        # Other common words
        'year', 'month', 'week', 'day', 'time', 'period',
        'first', 'second', 'third', 'last', 'next', 'previous',
        'table', 'figure', 'page', 'section', 'chapter',
        'al', 'et', 'and', 'or', 'but', 'nor', 'yet', 'so'
    }

    def __init__(self, use_embeddings: bool = True, enable_spacy: bool = True,
                 similarity_threshold: float = 0.5, log_file: str = 'citation_analyzer.log',
                 log_level=logging.INFO):
        """
        Initialize the hybrid citation analyzer.

        Args:
            use_embeddings: If True, uses sentence-transformers for semantic similarity
            enable_spacy: If True, uses spaCy for better sentence segmentation
            similarity_threshold: Minimum cosine similarity (0-1) to connect citations in network
            log_file: Path to log file (default: 'citation_analyzer.log')
            log_level: Logging level (default: logging.INFO)
        """
        # Setup logger
        global logger
        logger = setup_logger(log_file, log_level)
        logger.info("=" * 80)
        logger.info("Initializing Hybrid Semantic Citation Analyzer")
        logger.info("=" * 80)
        # Citation patterns (improved to reduce false positives)
        self.citation_patterns = {
            "doi": r'(?:https?://)?(?:dx\.)?doi\.org/[^\s,;)]+',
            "arxiv": r'arXiv:\d{4}\.\d{4,5}(?:v\d+)?',
            "url": r'https?://[^\s<>"()]+',
            "numbered": r'\[\d+(?:\s*[,;-]\s*\d+)*\]',
            # More strict author-year: requires at least 2 chars in name
            "author_year": r'\(([A-Z][a-z]{1,}(?: (?:and |& )?[A-Z][a-z]{1,})*(?:\s+et\s+al\.)?),?\s+(\d{4}[a-z]?)\)',
            # More strict Harvard: requires at least 2 chars
            "harvard": r'([A-Z][a-z]{1,}(?:\s+(?:and|&)\s+[A-Z][a-z]{1,})*)\s+\((\d{4}[a-z]?)\)',
            # More strict inline: requires at least 2 chars
            "inline": r'([A-Z][a-z]{1,})(?:\s+et\s+al\.?)?\s+(\d{4}[a-z]?)(?=\s|[,;.]|$)'
        }

        # Year extraction pattern with boundaries
        self.year_pattern = r'\b(19\d{2}|20\d{2})\b'
        
        # False positive patterns for year filtering
        self.year_false_positive_patterns = [
            r'\d{4}\s*[-–]\s*\d{4}',  # Range: "2020-2024"
            r'\d{4}\s*bit',            # "2048 bit"
            r'version\s*\d{4}',        # "version 2024"
            r'port\s*\d{4}',           # "port 8080"
            r'page\s*\d{4}',           # "page 1987"
            r'n\s*=\s*\d{4}',          # "n=2020"
            r'\d{4}\s*x\s*\d{4}',      # "1920x1080"
            r'ISO\s*\d{4}',            # "ISO 9001"
        ]

        # Stance/Sentiment Detection Signals
        self.stance_signals = {
            'supporting': [
                'confirms', 'supports', 'validates', 'verifies', 'corroborates',
                'agrees with', 'consistent with', 'aligns with', 'demonstrates',
                'proves', 'shows', 'establishes', 'reinforces', 'substantiates',
                'in line with', 'in agreement with'
            ],
            'refuting': [
                'contradicts', 'refutes', 'challenges', 'disputes', 'opposes',
                'questions', 'undermines', 'disproves', 'conflicts with',
                'contrary to', 'disagrees', 'fails to', 'unable to', 'incorrect',
                'flawed', 'problematic'
            ],
            'contrasting': [
                'however', 'although', 'whereas', 'while', 'but', 'nevertheless',
                'in contrast', 'on the other hand', 'conversely', 'unlike',
                'different from', 'differs from', 'alternatively', 'instead'
            ],
            'neutral': [
                'reports', 'describes', 'discusses', 'presents', 'reviews',
                'examines', 'investigates', 'analyzes', 'studies', 'explores',
                'notes', 'observes', 'mentions', 'states'
            ]
        }

        # Citation Purpose Classification Signals
        self.purpose_signals = {
            'background': [
                'background', 'previously', 'earlier work', 'established',
                'well-known', 'traditional', 'historical', 'foundational',
                'seminal', 'pioneering', 'literature review', 'prior research',
                'classic work', 'landmark study'
            ],
            'methodology': [
                'method', 'approach', 'technique', 'algorithm', 'procedure',
                'protocol', 'framework', 'model', 'implementation', 'using',
                'adopted', 'applied', 'following', 'based on', 'adapted from',
                'we use', 'we employ', 'we follow'
            ],
            'comparison': [
                'compared to', 'comparison with', 'versus', 'vs', 'relative to',
                'in contrast to', 'unlike', 'similar to', 'outperforms',
                'better than', 'worse than', 'benchmark', 'baseline',
                'state-of-the-art', 'competitive with'
            ],
            'theory': [
                'theory', 'theoretical', 'framework', 'concept', 'principle',
                'hypothesis', 'assumption', 'axiom', 'postulate', 'proposition',
                'conjecture', 'theorem', 'lemma'
            ],
            'results': [
                'results', 'findings', 'outcomes', 'observations', 'evidence',
                'demonstrates', 'shows', 'indicates', 'reveals', 'suggests',
                'found that', 'observed that', 'concluded that'
            ],
            'data': [
                'dataset', 'data', 'corpus', 'benchmark', 'collection',
                'database', 'repository', 'resource', 'sample', 'training set',
                'test set', 'validation set'
            ]
        }

        # Relationship keywords for co-citation detection
        self.relationship_keywords = {
            'extends': [
                'extends', 'builds on', 'expands', 'enhances', 'improves upon',
                'develops further', 'advances', 'refines'
            ],
            'contradicts': [
                'contradicts', 'refutes', 'challenges', 'disputes', 'opposes',
                'challenges', 'disputes', 'opposes', 'questions'
            ],
            'parallel': [
                'and', 'both', 'similarly', 'also', 'likewise', 'comparable to',
                'as well as', 'together with', 'along with'
            ],
            'supports': [
                'confirms', 'supports', 'validates', 'verifies', 'corroborates',
                'agrees with', 'consistent with', 'aligns with'
            ]
        }

        self.use_embeddings = use_embeddings
        self.enable_spacy = enable_spacy
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        self.nlp = None

        # Performance tracking
        self.timing_stats = {}
        
        # Initialize spaCy
        if self.enable_spacy:
            self._load_spacy()
        
        # Initialize embedding model
        if use_embeddings:
            self._load_embedding_model()
        
        logger.info(f"Configuration: embeddings={self.use_embeddings}, spacy={self.enable_spacy}")
        logger.info("Analyzer initialized successfully")
        logger.info("-" * 80)

    def _load_spacy(self):
        """Load spaCy model for better sentence segmentation."""
        try:
            import spacy
            logger.info("Loading spaCy model 'en_core_web_sm'...")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model loaded successfully")
        except ImportError:
            logger.warning("⚠️ spaCy not installed. Install: pip install spacy && python -m spacy download en_core_web_sm")
            logger.warning("⚠️ Falling back to basic sentence splitting")
            self.enable_spacy = False
        except OSError:
            logger.warning("⚠️ spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            logger.warning("⚠️ Falling back to basic sentence splitting")
            self.enable_spacy = False

    def _load_embedding_model(self):
        """Lazy load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence embedding model (one-time download ~80MB)...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Embedding model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
            logger.warning("Falling back to co-occurrence analysis only")
            self.use_embeddings = False
        except Exception as e:
            logger.warning(f"Error loading embedding model: {e}")
            logger.warning("Falling back to co-occurrence analysis only")
            self.use_embeddings = False

    def is_valid_citation_reference(self, text: str) -> bool:
        """
        Check if text is a valid citation reference (not a paper title/journal name).
        
        CRITICAL FIX: This filters out entries like "Computers 2025", "Access 2022"
        and keeps only valid citation references.
        """
        text = str(text).strip()
        
        # Valid patterns
        # Pattern 1: Numbered references [1], [2], [1,2], etc.
        if re.match(r'^\[\d+(?:\s*[,;-]\s*\d+)*\]$', text):
            return True
        
        # Pattern 2: DOI
        if 'doi.org' in text.lower():
            return True
        
        # Pattern 3: arXiv
        if text.startswith('arXiv:'):
            return True
        
        # Pattern 4: Author-year (short form only - not long titles)
        if re.match(r'^[A-Z][a-z]+.*\d{4}[a-z]?$', text) and len(text) < 50:
            # Additional check: ensure it's not in stopwords
            first_word = text.split()[0].lower() if text.split() else ''
            if first_word not in self.AUTHOR_STOPWORDS:
                return True
        
        # Exclude anything that looks like a journal/paper title
        # These typically contain words like "Computers", "Journal", "Proceedings", etc.
        exclude_words = [
            'computers', 'journal', 'proceedings', 'conference', 
            'access', 'review', 'transactions', 'international',
            'science', 'nature', 'ieee', 'acm', 'springer',
            'elsevier', 'wiley', 'taylor', 'francis', 'communications',
            'letters', 'magazine', 'bulletin', 'quarterly', 'annual'
        ]
        
        text_lower = text.lower()
        if any(word in text_lower for word in exclude_words):
            return False
        
        return False  # If none of the patterns match, it's not valid

    def _is_valid_author_name(self, name: str) -> bool:
        """
        Validate if a string is likely a valid author name.
        
        CRITICAL FIX: Filters out common words, months, etc.
        """
        if not name:
            return False
        
        # Check against stopword list
        name_lower = name.lower().strip()
        if name_lower in self.AUTHOR_STOPWORDS:
            return False
        
        # Check each word in multi-word names
        words = name_lower.split()
        for word in words:
            if word in self.AUTHOR_STOPWORDS:
                return False
        
        # Reject if starts with common conjunctions/prepositions
        if name_lower.startswith(('and ', 'or ', 'the ', 'a ', 'an ')):
            return False
        
        # Must be at least 2 characters
        if len(name) < 2:
            return False
        
        # Check if looks like a page marker or table reference
        if re.match(r'^(page|table|figure|section)\s+\d+', name_lower):
            return False
        
        return True

    def _get_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with position tracking.
        Uses spaCy if available, falls back to regex.
        
        Returns:
            List of (sentence_text, start_pos, end_pos)
        """
        if self.enable_spacy and self.nlp:
            doc = self.nlp(text)
            return [(sent.text, sent.start_char, sent.end_char) for sent in doc.sents]
        else:
            # Fallback: basic sentence splitting with regex
            sentences = []
            sentence_pattern = r'[.!?]+\s+'
            last_end = 0
            
            for match in re.finditer(sentence_pattern, text):
                sentences.append((text[last_end:match.end()].strip(), last_end, match.end()))
                last_end = match.end()
            
            # Add last sentence
            if last_end < len(text):
                sentences.append((text[last_end:].strip(), last_end, len(text)))
            
            return sentences

    def _get_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into paragraphs with position tracking.
        
        Returns:
            List of (paragraph_text, start_pos, end_pos)
        """
        paragraphs = []
        para_pattern = r'\n\s*\n'
        last_end = 0
        
        for match in re.finditer(para_pattern, text):
            para_text = text[last_end:match.start()].strip()
            if para_text:
                paragraphs.append((para_text, last_end, match.start()))
            last_end = match.end()
        
        # Add last paragraph
        if last_end < len(text):
            para_text = text[last_end:].strip()
            if para_text:
                paragraphs.append((para_text, last_end, len(text)))
        
        # If no paragraphs found, treat entire text as one paragraph
        if not paragraphs:
            paragraphs = [(text.strip(), 0, len(text))]
        
        return paragraphs

    def _clean_context_text(self, text: str, max_length: int = 600) -> str:
        """
        Clean and format context text for better readability.
        
        CRITICAL FIX: Removes page headers and PDF artifacts
        """
        if not text:
            return "No context available"
        
        # Remove page headers (common patterns)
        text = re.sub(r'--Page \d+--', '', text)
        text = re.sub(r'\[END TABLE \d+\]', '', text)
        text = re.sub(r'\[TABLE \d+ Page \d+\]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Fix common PDF extraction issues
        text = text.replace('- ', '')
        text = text.replace('\ufeff', '')
        text = text.replace('\u200b', '')
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
            last_sentence_end = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
            
            if last_sentence_end > len(text) * 0.6:
                text = text[:last_sentence_end + 1]
            else:
                last_space = text.rfind(' ')
                if last_space > len(text) * 0.8:
                    text = text[:last_space] + '...'
                else:
                    text = text + '...'
        
        return text.strip()

    def _is_year_false_positive(self, year_match: str, context: str) -> bool:
        """Check if a year match is likely a false positive."""
        for pattern in self.year_false_positive_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        return False

    def _extract_references_section(self, text: str) -> Optional[str]:
        """
        Extract the references/bibliography section from the paper.

        Returns:
            References section text or None if not found
        """
        # Common section headers for references
        reference_headers = [
            r'\n\s*REFERENCES?\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*WORKS?\s+CITED\s*\n',
            r'\n\s*LITERATURE\s+CITED\s*\n',
            r'\n\s*CITED\s+LITERATURE\s*\n',
        ]

        # Try to find references section
        for header_pattern in reference_headers:
            match = re.search(header_pattern, text, re.IGNORECASE)
            if match:
                # Extract from this point to end (or to appendix if present)
                start_pos = match.end()

                # Look for section that might come after references
                end_markers = [
                    r'\n\s*APPENDIX',
                    r'\n\s*SUPPLEMENTARY',
                    r'\n\s*ACKNOWLEDGMENT',
                ]

                end_pos = len(text)
                for end_pattern in end_markers:
                    end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE)
                    if end_match:
                        end_pos = start_pos + end_match.start()
                        break

                references_text = text[start_pos:end_pos]
                logger.info(f"✓ Found references section: {len(references_text)} characters")
                return references_text

        logger.warning("⚠️ Could not find references section")
        return None

    def _extract_years_from_references(self, references_text: str) -> Dict[int, int]:
        """
        Extract years specifically from the references section.
        Much more reliable than extracting from main text.

        Returns:
            Dictionary mapping year -> count
        """
        year_counts = defaultdict(int)

        # Split references into individual entries
        # Common patterns: numbered [1], numbered without brackets, or paragraph breaks
        reference_entries = []

        # Try numbered format [1], [2], etc.
        numbered_refs = re.split(r'\[\d+\]', references_text)
        if len(numbered_refs) > 3:  # If we found multiple numbered refs
            reference_entries = numbered_refs[1:]  # Skip first empty entry
        else:
            # Try format: 1., 2., etc.
            numbered_refs = re.split(r'\n\s*\d+\.\s+', references_text)
            if len(numbered_refs) > 3:
                reference_entries = numbered_refs[1:]
            else:
                # Fall back to paragraph-based splitting
                reference_entries = [p.strip() for p in references_text.split('\n\n') if p.strip()]

        logger.info(f"Found {len(reference_entries)} reference entries")

        # Extract years from each reference entry
        for entry in reference_entries:
            # Look for 4-digit years in the entry
            year_matches = re.finditer(self.year_pattern, entry)

            for match in year_matches:
                year_str = match.group(1)
                year = int(year_str)

                # Validate year range (1900-2030)
                if 1900 <= year <= 2030:
                    # In references, years are typically publication years
                    # Check if it's not a page number or other false positive
                    context = entry[max(0, match.start()-20):min(len(entry), match.end()+20)]

                    # Exclude if it looks like a page range (e.g., "pp. 1950-1960")
                    if re.search(r'pp?\.\s*\d+-' + year_str, context, re.IGNORECASE):
                        continue

                    # Exclude if it's part of a DOI or URL
                    if 'doi.org' in context.lower() or 'http' in context.lower():
                        # Check if year is part of URL
                        if re.search(r'[./]\d{4}', context):
                            continue

                    year_counts[year] += 1
                    # Typically only one year per reference, so break after first valid year
                    break

        return dict(year_counts)

    def _extract_years_with_validation(self, text: str) -> Dict[int, int]:
        """
        Extract years from text with false positive filtering.
        IMPROVED: First tries to extract from references section, falls back to full text.

        Returns:
            Dictionary mapping year -> count
        """
        # 🌟 NEW: Try to extract from references section first
        references_section = self._extract_references_section(text)

        if references_section:
            logger.info("Using references section for year extraction")
            year_counts = self._extract_years_from_references(references_section)

            if year_counts:
                logger.info(f"✓ Extracted {len(year_counts)} unique years from references")
                return year_counts
            else:
                logger.warning("No years found in references section, falling back to full text")

        # Fallback: Original method (extract from full text)
        logger.info("Using full text scan for year extraction")
        year_counts = defaultdict(int)

        # Get sentences for context validation
        sentences = self._get_sentences(text)

        for sentence, start_pos, end_pos in sentences:
            # Find all year matches in this sentence
            for match in re.finditer(self.year_pattern, sentence):
                year_str = match.group(1)
                year = int(year_str)

                # Validate year range (1900-2030)
                if not (1900 <= year <= 2030):
                    continue

                # Check for false positives
                context_window = sentence[max(0, match.start()-50):min(len(sentence), match.end()+50)]
                if self._is_year_false_positive(year_str, context_window):
                    continue

                # Check if year appears near citation patterns
                near_citation = False
                for pattern_name, pattern in self.citation_patterns.items():
                    if pattern_name in ['doi', 'arxiv', 'url']:
                        continue
                    if re.search(pattern, context_window):
                        near_citation = True
                        break

                if near_citation:
                    year_counts[year] += 1

        return dict(year_counts)

    def _detect_stance(self, context: str) -> Tuple[str, float]:
        """
        Detect citation stance from context.
        
        Returns:
            (stance, confidence)
        """
        context_lower = context.lower()
        
        scores = {}
        for stance, keywords in self.stance_signals.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            if score > 0:
                scores[stance] = score
        
        if not scores:
            return 'neutral', 0.5
        
        max_stance = max(scores, key=scores.get)
        max_score = scores[max_stance]
        confidence = min(0.9, 0.5 + (max_score * 0.1))
        
        return max_stance, confidence

    def _detect_purpose(self, context: str) -> Tuple[str, float]:
        """
        Detect citation purpose from context.
        
        Returns:
            (purpose, confidence)
        """
        context_lower = context.lower()
        
        scores = {}
        for purpose, keywords in self.purpose_signals.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            if score > 0:
                scores[purpose] = score
        
        if not scores:
            return 'general', 0.5
        
        max_purpose = max(scores, key=scores.get)
        max_score = scores[max_purpose]
        confidence = min(0.9, 0.5 + (max_score * 0.1))
        
        return max_purpose, confidence

    def _extract_citations_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citations with their surrounding context.
        
        CRITICAL FIX: Proper context extraction using sentence boundaries
        """
        citations = []
        seen_citations = set()
        
        # Get sentences
        sentences = self._get_sentences(text)
        paragraphs = self._get_paragraphs(text)
        
        # Map positions to paragraph indices
        position_to_paragraph = {}
        for para_idx, (para_text, para_start, para_end) in enumerate(paragraphs):
            for pos in range(para_start, para_end):
                position_to_paragraph[pos] = para_idx
        
        # Extract citations from each pattern
        for pattern_name, pattern in self.citation_patterns.items():
            for match in re.finditer(pattern, text):
                citation_text = match.group(0)
                position = match.start()
                
                # Skip if too short or already seen
                if len(citation_text) < 2:
                    continue
                
                # Additional validation for author-year patterns
                if pattern_name in ['author_year', 'harvard', 'inline']:
                    # Extract author name
                    if pattern_name == 'author_year':
                        author_name = match.group(1)
                    else:
                        author_name = match.group(1) if match.groups() else citation_text.split('(')[0].strip()
                    
                    # Validate author name
                    if not self._is_valid_author_name(author_name):
                        continue
                
                # Normalize citation for deduplication
                normalized_id = self._normalize_citation(citation_text, pattern_name)
                if normalized_id in seen_citations:
                    continue
                seen_citations.add(normalized_id)
                
                # Find sentence containing this citation
                containing_sentence = None
                for sent, sent_start, sent_end in sentences:
                    if sent_start <= position < sent_end:
                        containing_sentence = sent
                        break
                
                if not containing_sentence:
                    containing_sentence = text[max(0, position-200):min(len(text), position+200)]
                
                # Clean context
                context = self._clean_context_text(containing_sentence)
                
                # Detect stance and purpose
                stance, stance_conf = self._detect_stance(context)
                purpose, purpose_conf = self._detect_purpose(context)
                
                # Get paragraph index
                paragraph_idx = position_to_paragraph.get(position, 0)
                
                citations.append({
                    'text': citation_text,
                    'normalized_id': normalized_id,
                    'pattern_type': pattern_name,
                    'position': position,
                    'context': context,
                    'stance': stance,
                    'stance_confidence': stance_conf,
                    'purpose': purpose,
                    'purpose_confidence': purpose_conf,
                    'paragraph_idx': paragraph_idx
                })
        
        return citations

    def _normalize_citation(self, citation: str, pattern_type: str) -> str:
        """Normalize citation text for deduplication."""
        if pattern_type == 'numbered':
            # Extract just the numbers
            numbers = re.findall(r'\d+', citation)
            return f"[{','.join(numbers)}]"
        elif pattern_type in ['doi', 'arxiv', 'url']:
            return citation.strip()
        else:
            # Author-year normalization
            return citation.strip()

    def _compute_semantic_similarity(self, citations: List[Dict]) -> float:
        """Compute semantic similarity between citation contexts."""
        if not self.use_embeddings or not self.embedding_model:
            return 0.0
        
        contexts = [cite['context'] for cite in citations]
        if len(contexts) < 2:
            return 0.0
        
        try:
            embeddings = self.embedding_model.encode(contexts)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(embeddings)
            avg_similarity = (similarities.sum() - len(similarities)) / (len(similarities) * (len(similarities) - 1))
            return float(avg_similarity)
        except:
            return 0.0

    def _build_hybrid_network(self, text: str, citations: List[Dict]) -> Dict:
        """
        Build citation network combining co-occurrence and semantic similarity.
        
        CRITICAL FIX: Proper citation-to-citation relationships, no star topology
        """
        G = nx.Graph()
        
        # Add all citations as nodes
        for cite in citations:
            if G.has_node(cite['normalized_id']):
                # Update existing node
                G.nodes[cite['normalized_id']]['contexts'].append(cite['context'])
                G.nodes[cite['normalized_id']]['stances'].append(cite['stance'])
                G.nodes[cite['normalized_id']]['purposes'].append(cite['purpose'])
            else:
                # Add new node
                G.add_node(
                    cite['normalized_id'],
                    pattern_type=cite['pattern_type'],
                    contexts=[cite['context']],
                    stances=[cite['stance']],
                    purposes=[cite['purpose']]
                )
        
        # Get sentences for co-occurrence detection
        sentences = self._get_sentences(text)
        
        # Find co-occurring citations (same sentence)
        sentence_citations = defaultdict(list)
        for cite in citations:
            for sent_idx, (sent, sent_start, sent_end) in enumerate(sentences):
                if sent_start <= cite['position'] < sent_end:
                    sentence_citations[sent_idx].append(cite)
                    break
        
        # Build co-occurrence pairs
        co_occurrence_pairs = defaultdict(lambda: {'weight': 0, 'contexts': []})
        for sent_idx, sent_cites in sentence_citations.items():
            if len(sent_cites) > 1:
                for i in range(len(sent_cites)):
                    for j in range(i + 1, len(sent_cites)):
                        cite1 = sent_cites[i]
                        cite2 = sent_cites[j]
                        
                        # Create ordered pair
                        pair = tuple(sorted([cite1['normalized_id'], cite2['normalized_id']]))
                        co_occurrence_pairs[pair]['weight'] += 1
                        co_occurrence_pairs[pair]['contexts'].append(cite1['context'])
        
        # Add co-occurrence edges to graph
        edge_details_map = {}
        citations_in_context = []
        
        for (cite1, cite2), data in co_occurrence_pairs.items():
            weight = min(data['weight'], 5)  # Cap weight at 5
            
            G.add_edge(cite1, cite2, 
                      weight=weight,
                      relationship='co-occurrence')
            
            # Store edge details for export
            edge_details_map[(cite1, cite2)] = {
                'from': cite1,
                'to': cite2,
                'relationship': 'co-occurrence',
                'weight': weight,
                'from_stance': G.nodes[cite1]['stances'][0] if G.nodes[cite1]['stances'] else 'neutral',
                'to_stance': G.nodes[cite2]['stances'][0] if G.nodes[cite2]['stances'] else 'neutral',
                'from_purpose': G.nodes[cite1]['purposes'][0] if G.nodes[cite1]['purposes'] else 'general',
                'to_purpose': G.nodes[cite2]['purposes'][0] if G.nodes[cite2]['purposes'] else 'general'
            }
        
        # Add semantic similarity edges
        if self.use_embeddings and self.embedding_model and len(citations) > 1:
            try:
                contexts = [cite['context'] for cite in citations]
                embeddings = self.embedding_model.encode(contexts)
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(embeddings)
                
                for i in range(len(citations)):
                    for j in range(i + 1, len(citations)):
                        similarity = similarity_matrix[i][j]

                        if similarity > self.similarity_threshold:  # Threshold for semantic similarity
                            cite1 = citations[i]
                            cite2 = citations[j]
                            
                            pair = tuple(sorted([cite1['normalized_id'], cite2['normalized_id']]))
                            
                            # If already has co-occurrence edge, increase weight
                            if G.has_edge(cite1['normalized_id'], cite2['normalized_id']):
                                G[cite1['normalized_id']][cite2['normalized_id']]['weight'] += 1
                                G[cite1['normalized_id']][cite2['normalized_id']]['relationship'] = 'hybrid'
                                G[cite1['normalized_id']][cite2['normalized_id']]['similarity_score'] = float(similarity)
                            else:
                                # Add new semantic edge
                                weight = 2
                                relationship_type = f"semantic_similar (sim: {similarity:.2f})"
                                
                                G.add_edge(
                                    cite1['normalized_id'],
                                    cite2['normalized_id'],
                                    weight=weight,
                                    relationship=relationship_type,
                                    similarity_score=float(similarity)
                                )
                                
                                # Store edge details
                                edge_details_map[pair] = {
                                    'from': cite1['normalized_id'],
                                    'to': cite2['normalized_id'],
                                    'relationship': relationship_type,
                                    'weight': weight,
                                    'similarity': float(similarity),
                                    'from_stance': cite1['stance'],
                                    'to_stance': cite2['stance'],
                                    'from_purpose': cite1['purpose'],
                                    'to_purpose': cite2['purpose']
                                }
                                
                                citations_in_context.append({
                                    'from': cite1['normalized_id'],
                                    'to': cite2['normalized_id'],
                                    'relationship': relationship_type,
                                    'weight': weight,
                                    'similarity': float(similarity),
                                    'from_stance': cite1['stance'],
                                    'to_stance': cite2['stance'],
                                    'from_purpose': cite1['purpose'],
                                    'to_purpose': cite2['purpose']
                                })
            except Exception as e:
                logger.warning(f"Error computing semantic similarity: {e}")
        
        # Calculate network metrics
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        
        network_metrics = {
            'node_count': node_count,
            'edge_count': edge_count,
            'density': nx.density(G) if node_count > 1 else 0,
            'average_degree': sum(dict(G.degree()).values()) / node_count if node_count > 0 else 0
        }
        
        # Compute centrality
        if node_count > 0:
            centrality = nx.degree_centrality(G)
            top_citations = sorted(
                [{'citation': node, 'connections': G.degree(node), 'centrality': centrality[node]}
                 for node in G.nodes()],
                key=lambda x: x['connections'],
                reverse=True
            )[:5]
            network_metrics['top_citations'] = top_citations
        
        # Community detection
        communities = []
        if node_count > 1 and edge_count > 0:
            try:
                communities_generator = nx.community.greedy_modularity_communities(G)
                communities = [list(community) for community in communities_generator]
                network_metrics['num_communities'] = len(communities)
            except:
                network_metrics['num_communities'] = 0
        
        # Build final network data structure (WITHOUT context in edges)
        network_data = {
            'nodes': [
                {
                    'id': node,
                    'label': node,
                    'degree': G.degree(node),
                    'pattern_type': G.nodes[node].get('pattern_type', 'unknown'),
                    'num_contexts': len(G.nodes[node].get('contexts', [])),
                    'dominant_stance': max(set(G.nodes[node].get('stances', ['neutral'])),
                                          key=G.nodes[node].get('stances', ['neutral']).count),
                    'dominant_purpose': max(set(G.nodes[node].get('purposes', ['general'])),
                                           key=G.nodes[node].get('purposes', ['general']).count)
                }
                for node in G.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': G[u][v].get('weight', 1),
                    'relationship': G[u][v].get('relationship', 'unknown'),
                    'similarity': G[u][v].get('similarity_score', None),
                    'source_stance': edge_details_map.get(tuple(sorted([u, v])), {}).get('from_stance', 'unknown'),
                    'target_stance': edge_details_map.get(tuple(sorted([u, v])), {}).get('to_stance', 'unknown'),
                    'source_purpose': edge_details_map.get(tuple(sorted([u, v])), {}).get('from_purpose', 'unknown'),
                    'target_purpose': edge_details_map.get(tuple(sorted([u, v])), {}).get('to_purpose', 'unknown')
                }
                for u, v in G.edges()
            ],
            'metrics': network_metrics,
            'communities': communities,
            'citations_in_context': citations_in_context
        }
        
        # ⭐ CRITICAL FIX: Filter edges to remove non-citation entries
        logger.info(f"Edges before filtering: {len(network_data['edges'])}")
        network_data['edges'] = [
            edge for edge in network_data['edges']
            if self.is_valid_citation_reference(edge['source']) 
            and self.is_valid_citation_reference(edge['target'])
        ]
        logger.info(f"Edges after filtering: {len(network_data['edges'])}")
        
        # Also filter nodes
        valid_node_ids = set()
        for edge in network_data['edges']:
            valid_node_ids.add(edge['source'])
            valid_node_ids.add(edge['target'])
        
        network_data['nodes'] = [
            node for node in network_data['nodes']
            if node['id'] in valid_node_ids
        ]
        
        # Recalculate metrics
        network_data['metrics']['node_count'] = len(network_data['nodes'])
        network_data['metrics']['edge_count'] = len(network_data['edges'])
        
        return network_data

    def extract_citations(self, text: str) -> Dict[str, Any]:
        """
        Extract citations with hybrid semantic and co-occurrence analysis.
        
        Args:
            text: Research paper text
            
        Returns:
            Dictionary with comprehensive citation analysis
        """
        start_time = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("Starting hybrid citation analysis")
        logger.info("=" * 80)
        logger.info(f"Input text length: {len(text)} characters")
        
        # Stage 1: Extract citations with full context
        stage1_start = time.time()
        logger.info("\n[Stage 1/3] Extracting citations with context...")
        citations = self._extract_citations_with_context(text)
        self.timing_stats['extraction'] = time.time() - stage1_start
        logger.info(f"✓ Extracted {len(citations)} valid citations in {self.timing_stats['extraction']:.3f}s")
        
        # Stage 2: Extract years with validation
        stage2_start = time.time()
        logger.info("\n[Stage 2/3] Extracting year distribution with validation...")
        year_distribution = self._extract_years_with_validation(text)
        self.timing_stats['year_extraction'] = time.time() - stage2_start
        logger.info(f"✓ Found {len(year_distribution)} unique years in {self.timing_stats['year_extraction']:.3f}s")
        
        # Stage 3: Build hybrid network
        stage3_start = time.time()
        logger.info("\n[Stage 3/3] Building hybrid citation network...")
        network = self._build_hybrid_network(text, citations)
        self.timing_stats['network_building'] = time.time() - stage3_start
        logger.info(f"✓ Built network with {network['metrics']['node_count']} nodes and {network['metrics']['edge_count']} edges in {self.timing_stats['network_building']:.3f}s")
        
        # Compile statistics
        citation_counts = defaultdict(int)
        stance_distribution = defaultdict(int)
        purpose_distribution = defaultdict(int)
        
        for cite in citations:
            citation_counts[cite['pattern_type']] += 1
            stance_distribution[cite['stance']] += 1
            purpose_distribution[cite['purpose']] += 1
        
        # Create detailed citation table (WITH context)
        citation_details = [
            {
                'citation': cite['normalized_id'],
                'type': cite['pattern_type'],
                'stance': cite['stance'],
                'stance_confidence': cite['stance_confidence'],
                'purpose': cite['purpose'],
                'purpose_confidence': cite['purpose_confidence'],
                'context': cite['context'],  # Context kept in details
                'position': cite['position'],
                'paragraph_idx': cite['paragraph_idx']
            }
            for cite in citations
        ]
        
        total_time = time.time() - start_time
        self.timing_stats['total_time'] = total_time
        
        # Build final result
        result = {
            'total_count': len(citations),
            'unique_count': len(set(cite['normalized_id'] for cite in citations)),
            'year_distribution': year_distribution,
            'stance_distribution': dict(stance_distribution),
            'purpose_distribution': dict(purpose_distribution),
            'network': network,
            'citation_details': citation_details,
            'method': 'hybrid_semantic_cooccurrence',
            'features': {
                'embeddings': self.use_embeddings,
                'spacy': self.enable_spacy,
                'co_occurrence': True,
                'semantic_similarity': self.use_embeddings,
                'false_positive_filtering': True
            },
            'timing': self.timing_stats
        }
        
        # Add counts by type
        for ctype, count in citation_counts.items():
            result[f'{ctype}_count'] = count
        
        # Log summary
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total citations found: {result['total_count']}")
        logger.info(f"Unique citations: {result['unique_count']}")
        logger.info(f"Network nodes: {network['metrics']['node_count']}")
        logger.info(f"Network edges: {network['metrics']['edge_count']}")
        logger.info(f"Network density: {network['metrics']['density']:.3f}")
        logger.info(f"Total processing time: {total_time:.3f}s")
        
        # Log citation type breakdown
        logger.info("\nCitation Types:")
        for ctype, count in citation_counts.items():
            if count > 0:
                logger.info(f"  • {ctype}: {count}")
        
        # Log stance distribution
        if stance_distribution:
            logger.info("\nStance Distribution:")
            for stance, count in sorted(stance_distribution.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  • {stance}: {count}")
        
        # Log purpose distribution
        if purpose_distribution:
            logger.info("\nPurpose Distribution:")
            for purpose, count in sorted(purpose_distribution.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  • {purpose}: {count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Analysis completed successfully")
        logger.info("=" * 80 + "\n")
        
        return result

    def get_citation_insights(self, network_data: Dict) -> str:
        """Generate human-readable insights from the citation network."""
        insights = []
        metrics = network_data['metrics']

        insights.append("📊 Network Overview:")
        insights.append(f"  • {metrics['node_count']} unique citations found")
        insights.append(f"  • {metrics['edge_count']} relationships identified")
        insights.append(f"  • Network density: {metrics['density']:.2%}")
        insights.append(f"  • Average connections per citation: {metrics['average_degree']:.1f}")

        if metrics.get('num_communities', 0) > 0:
            insights.append(f"\n🔍 Topic Clusters:")
            insights.append(f"  • Found {metrics['num_communities']} research topic clusters")

        if 'top_citations' in metrics and metrics['top_citations']:
            insights.append(f"\n⭐ Most Influential Citations:")
            for cite_info in metrics['top_citations'][:3]:
                insights.append(f"  • {cite_info['citation']}: {cite_info['connections']} connections")

        if 'citations_in_context' in network_data and network_data['citations_in_context']:
            insights.append(f"\n📑 Sample Citation Relationships:")
            for ctx in network_data['citations_in_context'][:3]:
                rel_type = ctx['relationship']
                sim_score = f" (similarity: {ctx['similarity']:.2f})" if ctx.get('similarity') else ""
                insights.append(f"  • {ctx['from']} → {ctx['to']}")
                insights.append(f"    Type: {rel_type}{sim_score}")
                insights.append(f"    Stances: {ctx['from_stance']} ↔ {ctx['to_stance']}")

        return "\n".join(insights)

    def get_stance_purpose_summary(self, result: Dict) -> str:
        """Generate summary of stance and purpose distributions."""
        summary = []
        
        summary.append("📈 Citation Stance Distribution:")
        stance_dist = result.get('stance_distribution', {})
        total_stances = sum(stance_dist.values())
        for stance, count in sorted(stance_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_stances * 100) if total_stances > 0 else 0
            summary.append(f"  • {stance.capitalize()}: {count} ({percentage:.1f}%)")
        
        summary.append("\n🎯 Citation Purpose Distribution:")
        purpose_dist = result.get('purpose_distribution', {})
        total_purposes = sum(purpose_dist.values())
        for purpose, count in sorted(purpose_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_purposes * 100) if total_purposes > 0 else 0
            summary.append(f"  • {purpose.capitalize()}: {count} ({percentage:.1f}%)")
        
        return "\n".join(summary)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    log_file = 'citation_analyzer_test.log'
    test_logger = setup_logger(log_file, logging.INFO)
    
    sample_text = """
    Recent work by Smith et al. (2020) has shown promising results in machine learning.
    Building on this methodology, Jones (2021) extended the approach with deep learning.
    However, Brown et al. (2022) contradicts these findings and questions the validity.
    Similar methods were used in previous research [1] and [2].
    The background literature demonstrates that Miller (2019) established the theoretical framework.
    For comparison, our results outperform the baseline presented in Davis (2023).
    We adopted the data collection protocol from Wilson et al. (2018).
    According to Taylor (2017), the results align with previous work in this domain.
    """
    
    test_logger.info("=" * 80)
    test_logger.info("🚀 FIXED Hybrid Semantic Citation Analyzer - TEST RUN")
    test_logger.info("=" * 80)
    
    # Initialize analyzer
    test_logger.info("\nInitializing analyzer with all features...")
    analyzer = HybridSemanticCitationAnalyzer(
        use_embeddings=True,
        enable_spacy=True,
        log_file=log_file
    )
    
    # Extract citations
    test_logger.info("\n📊 Analyzing citations...")
    results = analyzer.extract_citations(sample_text)
    
    # Display results
    test_logger.info(f"\n✓ Total citations found: {results['total_count']}")
    test_logger.info(f"✓ Unique citations: {results['unique_count']}")
    test_logger.info(f"✓ Method: {results['method']}")
    test_logger.info(f"✓ Features: {results['features']}")
    
    # Display year distribution
    if results['year_distribution']:
        test_logger.info("\n📅 Year Distribution:")
        for year, count in sorted(results['year_distribution'].items()):
            test_logger.info(f"  • {year}: {count} citations")
    
    # Display stance and purpose
    test_logger.info(f"\n{analyzer.get_stance_purpose_summary(results)}")
    
    # Display network insights
    test_logger.info(f"\n{analyzer.get_citation_insights(results['network'])}")
    
    # Performance
    test_logger.info("\n⏱️  Performance:")
    for stage, duration in results['timing'].items():
        test_logger.info(f"  • {stage}: {duration:.3f}s")
    
    test_logger.info("\n" + "=" * 80)
    test_logger.info("✅ Analysis complete! Check log file: " + log_file)
    test_logger.info("=" * 80)
    
    # Also print to console where to find the log
    print(f"\n✅ Analysis complete! Results logged to: {log_file}")
    print(f"   Total citations: {results['total_count']}")
    print(f"   Processing time: {results['timing']['total_time']:.3f}s")