import os
import shutil
from datetime import datetime, timedelta
from .config import settings
from .logger import logger

class FileManager:
    def __init__(self):
        self.storage_path = settings.PDF_STORAGE_PATH
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_pdf(self, pdf_content: bytes, filename: str) -> str:
        """Save PDF content to storage."""
        try:
            filepath = os.path.join(self.storage_path, filename)
            with open(filepath, 'wb') as f:
                f.write(pdf_content)
            logger.info(f"PDF saved successfully: {filename}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving PDF {filename}: {str(e)}")
            raise
    
    def cleanup_old_files(self):
        """Remove PDFs older than MAX_PDF_AGE_DAYS."""
        try:
            cutoff_date = datetime.now() - timedelta(days=settings.MAX_PDF_AGE_DAYS)
            for filename in os.listdir(self.storage_path):
                filepath = os.path.join(self.storage_path, filename)
                if os.path.getctime(filepath) < cutoff_date.timestamp():
                    os.remove(filepath)
                    logger.info(f"Removed old PDF: {filename}")
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")
            raise

file_manager = FileManager() 