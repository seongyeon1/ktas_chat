#!pip install pymupdf
import fitz  # PyMuPDF

def split_pdf(input_pdf_path, output_pdf_path):
    # 원본 PDF 파일을 열기
    pdf = fitz.open(input_pdf_path)
    
    # 새로운 PDF 파일을 만들기
    new_pdf = fitz.open()

    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)
        rect = page.rect

        # 가운데를 기준으로 두 부분으로 나누기
        left_rect = fitz.Rect(rect.x0, rect.y0, rect.width / 2, rect.y1)
        right_rect = fitz.Rect(rect.width / 2, rect.y0, rect.x1, rect.y1)

        # 왼쪽 부분을 새 페이지에 추가
        left_page = new_pdf.new_page(width=left_rect.width, height=left_rect.height)
        left_page.show_pdf_page(left_rect, pdf, page_num, clip=left_rect)

        # 오른쪽 부분을 새 페이지에 추가
        right_page = new_pdf.new_page(width=right_rect.width, height=right_rect.height)
        right_page.show_pdf_page(right_rect, pdf, page_num, clip=right_rect)

    # 분할된 페이지들을 포함한 새로운 PDF 파일 저장
    new_pdf.save(output_pdf_path)
    new_pdf.close()
    pdf.close()

# 사용 예시
input_pdf_path = "input.pdf"  # 원본 PDF 파일 경로
output_pdf_path = "output_split.pdf"  # 저장할 PDF 파일 경로
split_pdf(input_pdf_path, output_pdf_path)