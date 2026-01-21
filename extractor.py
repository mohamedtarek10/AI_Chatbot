import fitz
import numpy as np
import pytesseract
from typing import List, Optional, Dict, Union
from pathlib import Path
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class PyMuPDFimageReader(BaseReader):
    """Read PDF files using PyMuPDF library with OCR for image-based text."""

    def __init__(self, extract_images: bool = False):
        """Initialize the reader with an option to extract images and process them with OCR."""
        self.extract_images = extract_images

    def _extract_images_from_page(self, doc: fitz.Document, page: fitz.Page) -> str:
        """Extract images from a page and use Tesseract OCR for text extraction."""
        if not self.extract_images:
            return ""

        img_list = page.get_images(full=True)
        images = []

        for img in img_list:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            try:
                # Always convert image to RGB to ensure consistent data
                if pix.n != 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                images.append(image_array)

            except Exception as e:
                print(f"Skipping image on page {page.number + 1} due to error: {e}")
                continue  # Skip problematic images

            # Check for number of channels and handle accordingly
            if pix.n == 1:  # Grayscale image
                print(f"Grayscale image on page {page.number + 1}")
                image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            elif pix.n == 3:  # RGB image
                print(f"RGB image on page {page.number + 1}")
                image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            elif pix.n == 4:  # CMYK image, convert to RGB
                print(f"CMYK image on page {page.number + 1}")
                pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB
                image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            else:
                print(f"Unknown image type on page {page.number + 1}. Skipping.")
                continue  # Skip unsupported image formats

            images.append(image_array)

        # Extract text using Tesseract OCR
        return self.extract_text_using_tesseract(images)



    def extract_text_using_tesseract(self, images: List[np.ndarray]) -> str:
        """Extract text from images using Tesseract OCR."""
        extracted_text = ""

        for img in images:
            text = pytesseract.image_to_string(img,lang="ara+eng")  # Extract text from image using Tesseract OCR
            if text.strip():  # Only append non-empty text
                extracted_text += text.strip() + "\n"

        return extracted_text.strip()
    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
        fs=None  # <-- added here to avoid the error
    ) -> List[Document]:
        """Load list of documents from a PDF file, including image-based OCR."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info, fs=fs)


    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
        fs=None  # <-- added here too
    ) -> List[Document]:
        """Load data and extract text and image-based OCR."""

        # Optional: support reading from S3 using fsspec if fs is passed
        if fs is not None and str(file_path).startswith("s3://"):
            with fs.open(file_path, "rb") as f:
                doc = fitz.open("pdf", f.read())
        else:
            doc = fitz.open(file_path)


        if extra_info is None:
            extra_info = {}

        if metadata:
            extra_info["total_pages"] = len(doc)
            extra_info["file_path"] = str(file_path)

        documents = []

        for page in doc:
            # Extract text from the page
            text = page.get_text()

            # Extract text from images using Tesseract OCR
            image_text = self._extract_images_from_page(doc, page)
            combined_text = text + "\n" + image_text if image_text else text

            # Create a Document object for the page
            documents.append(
                Document(
                    text=combined_text,
                    extra_info=dict(extra_info, **{"source": f"Page {page.number + 1}"}),
                )
            )

        return documents