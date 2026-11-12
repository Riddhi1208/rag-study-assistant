from io import BytesIO
import unittest

from pypdf import PdfWriter

from ingest import load_uploaded_documents


class UploadedDocumentTests(unittest.TestCase):
    def test_reports_a_pdf_without_extractable_text(self):
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf_buffer = BytesIO()
        writer.write(pdf_buffer)

        documents, files_without_text = load_uploaded_documents(
            [("blank.pdf", pdf_buffer.getvalue())]
        )

        self.assertEqual(documents, [])
        self.assertEqual(files_without_text, ["blank.pdf"])

    def test_rejects_an_invalid_pdf_with_its_filename(self):
        with self.assertRaisesRegex(ValueError, "broken.pdf"):
            load_uploaded_documents([("broken.pdf", b"not a PDF")])


if __name__ == "__main__":
    unittest.main()
