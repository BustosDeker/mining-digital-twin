from backend.reports.pdf_generator import generate_training_report as generate_pdf_report
from backend.reports.pdf_generator import generate_training_report_to_memory
from backend.reports.word_generator import generate_training_report as generate_word_report

__all__ = ["generate_pdf_report", "generate_training_report_to_memory", "generate_word_report"]