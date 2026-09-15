"""
Bước 6: Xuất file PDF tóm tắt cuộc họp (PDF Generation)
---------------------------------------------------------
Đầu vào  : file JSON (truyền qua command line)
Đầu ra   : file PDF (truyền qua command line hoặc mặc định Meeting_Summary.pdf)

Cách dùng:
  python Generate_pdf.py input.json                        # xuất ra Meeting_Summary.pdf
  python Generate_pdf.py input.json -o output.pdf          # chỉ định file PDF đầu ra
  python Generate_pdf.py input.json -o out.pdf -t "Tiêu đề cuộc họp"

Tính năng:
  - Hỗ trợ 100% tiếng Việt có dấu qua font BeVietnamPro (Google Fonts).
  - Tự động tải font nếu máy chưa có.
  - Phân trang, header, footer đánh số trang tự động.
  - Lọc bỏ ký tự Markdown thô và các câu thoại tiếng Anh thừa của AI.
"""

import json
import os
import re
import sys
import argparse
import urllib.request
from fpdf import FPDF

# -------------------------------------------------
# Cấu hình mặc định
# -------------------------------------------------
DEFAULT_OUTPUT_PDF = "Meeting.pdf"

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fronts")
FONT_REGULAR = os.path.join(FONT_DIR, "BeVietnamPro-Regular.ttf")
FONT_BOLD    = os.path.join(FONT_DIR, "BeVietnamPro-Bold.ttf")

URL_REGULAR = "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Regular.ttf"
URL_BOLD    = "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Bold.ttf"


def get_font_paths() -> tuple[str, str]:
    """Tự động kiểm tra hoặc tải font BeVietnamPro chuẩn tiếng Việt"""
    # 1. Kiểm tra nếu font đã có sẵn trong thư mục dự án
    if os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD):
        return FONT_REGULAR, FONT_BOLD

    # 2. Kiểm tra các font hệ thống Linux / Kaggle
    system_candidates = [
        ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for reg, bld in system_candidates:
        if os.path.exists(reg) and os.path.exists(bld):
            return reg, bld

    # 3. Tải font BeVietnamPro trực tiếp từ Google Fonts CDN
    print(" Đang tải font tiếng Việt BeVietnamPro từ Google Fonts...")
    try:
        urllib.request.urlretrieve(URL_REGULAR, FONT_REGULAR)
        urllib.request.urlretrieve(URL_BOLD, FONT_BOLD)
        print(" Tải font thành công!")
        return FONT_REGULAR, FONT_BOLD
    except Exception as e:
        print(f" Không thể tải font qua mạng: {e}")
        return None, None


class MeetingPDF(FPDF):
    """Custom PDF class có Header và Footer chuyên nghiệp"""
    def __init__(self, font_name="VietFont", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_name = font_name

    def header(self):
        self.set_font(self.font_name, "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "AI MEETING ASSISTANT - BIÊN BẢN TÓM TẮT CUỘC HỌP", border="B", align="L")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_name, "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Trang {self.page_no()}/{{nb}}", align="C")


def clean_line(text: str) -> str:
    """Lọc bỏ ký tự Markdown và các câu tiếng Anh thừa của AI"""
    skip_phrases = [
        "okay, here's a summary", "okay, here's a summary",
        "would you like me to elaborate", "translation of the bullet points",
        "explanation of the summary", "here is a summary",
        "presented in bullet points", "broken down into"
    ]
    low = text.strip().lower()
    for phrase in skip_phrases:
        if phrase in low:
            return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"[\*_](.*?)[\*_]", r"\1", text)
    return text.strip()


def export_summary_to_pdf(
    input_json: str,
    output_pdf: str = DEFAULT_OUTPUT_PDF,
    meeting_title: str = "BIÊN BẢN CUỘC HỌP TỔNG HỢP",
) -> str:
    """
    Hàm xuất dữ liệu từ file JSON ra file PDF chuẩn tiếng Việt.

    Args:
        input_json: Đường dẫn tới file JSON đầu vào (bắt buộc).
        output_pdf: Đường dẫn file PDF đầu ra (mặc định: Meeting_Summary.pdf).
        meeting_title: Tiêu đề cuộc họp hiển thị trên PDF.

    Returns:
        Đường dẫn file PDF đã xuất.
    """
    # Kiểm tra file JSON đầu vào
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Không tìm thấy file JSON đầu vào: '{input_json}'")

    if not input_json.lower().endswith(".json"):
        raise ValueError(f"File đầu vào phải là file JSON (.json), nhận được: '{input_json}'")

    print(f" Đọc file JSON: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall_summary = data.get("overall_summary", "")
    per_speaker = data.get("per_speaker", {})
    speakers = data.get("speakers", [])

    # Chuẩn bị font chữ tiếng Việt
    reg_font, bold_font = get_font_paths()
    font_name = "VietFont"

    pdf = MeetingPDF(font_name=font_name, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)

    # Đăng ký font UTF-8
    if reg_font:
        pdf.add_font(font_name, "", reg_font)
        pdf.add_font(font_name, "B", bold_font if bold_font else reg_font)
    else:
        pdf.set_fallback_fonts(["Helvetica"])

    pdf.add_page()

    # 1. TIÊU ĐỀ CHÍNH
    pdf.set_font(font_name, "B", 16)
    pdf.set_text_color(24, 76, 120)  # Xanh dương đậm
    pdf.cell(0, 10, meeting_title, align="C")
    pdf.ln(10)

    # Thông tin người tham gia
    pdf.set_font(font_name, "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Người tham gia: {', '.join(speakers) if speakers else 'Không xác định'}", align="C")
    pdf.ln(6)

    # Đường kẻ ngang phân cách
    pdf.set_draw_color(24, 76, 120)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # 2. IN NỘI DUNG TỔNG QUAN
    lines = overall_summary.replace("\\n", "\n").split("\n")
    for raw_line in lines:
        line = clean_line(raw_line)
        if not line:
            continue

        # Tiêu đề mục (## ...)
        if raw_line.strip().startswith("##"):
            pdf.ln(3)
            pdf.set_font(font_name, "B", 12)
            pdf.set_text_color(24, 76, 120)
            pdf.cell(0, 7, line.replace("#", "").strip())
            pdf.ln(7)

        # Gạch đầu dòng (• ...)
        elif raw_line.strip().startswith(("*", "-", "•")):
            bullet_text = line.lstrip("*-• ").strip()
            pdf.set_font(font_name, "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.set_x(20)
            pdf.multi_cell(0, 5.5, f"•  {bullet_text}")
            pdf.ln(1)

        # Đoạn văn thông thường
        else:
            pdf.set_font(font_name, "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 5.5, line)
            pdf.ln(2)

    # 3. IN NỘI DUNG TỪNG SPEAKER
    if per_speaker:
        pdf.ln(4)
        pdf.set_font(font_name, "B", 12)
        pdf.set_text_color(24, 76, 120)
        pdf.cell(0, 7, "V. CHI TIẾT Ý KIẾN TỪNG NGƯỜI NÓI")
        pdf.ln(7)

        for sp, summary_text in per_speaker.items():
            pdf.set_font(font_name, "B", 10.5)
            pdf.set_text_color(40, 116, 166)
            pdf.cell(0, 6, f"Ý kiến của {sp}:")
            pdf.ln(6)

            sp_lines = summary_text.replace("\\n", "\n").split("\n")
            for sp_raw in sp_lines:
                sp_clean = clean_line(sp_raw)
                if not sp_clean:
                    continue
                pdf.set_font(font_name, "", 9.5)
                pdf.set_text_color(30, 30, 30)
                pdf.set_x(20)
                pdf.multi_cell(0, 5, f"•  {sp_clean.lstrip('*-• ')}")
                pdf.ln(1)
            pdf.ln(2)

    # Xuất file
    pdf.output(output_pdf)
    print(f"\n ĐÃ XUẤT FILE PDF TIẾNG VIỆT THÀNH CÔNG: '{output_pdf}'")
    return output_pdf


def export_transcript_to_pdf(
    input_json: str,
    output_pdf: str = "Transcript.pdf",
    meeting_title: str = "BIÊN BẢN CUỘC HỌP CHI TIẾT",
) -> str:
    """
    Xuất file formatted_transcript.json ra PDF theo thứ tự thời gian.

    Hiển thị tuần tự từng phát biểu theo đúng trình tự cuộc họp,
    KHÔNG gom theo speaker.

    Args:
        input_json: Đường dẫn tới file JSON transcript (bắt buộc).
        output_pdf: Đường dẫn file PDF đầu ra.
        meeting_title: Tiêu đề hiển thị trên PDF.

    Returns:
        Đường dẫn file PDF đã xuất.
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Không tìm thấy file JSON đầu vào: '{input_json}'")

    if not input_json.lower().endswith(".json"):
        raise ValueError(f"File đầu vào phải là file JSON (.json), nhận được: '{input_json}'")

    print(f" Đọc file transcript JSON: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not isinstance(segments, list):
        raise ValueError("File JSON transcript phải là một danh sách (list) các segment.")

    # Lấy danh sách speaker (không trùng lặp, giữ thứ tự xuất hiện)
    all_speakers = list(dict.fromkeys(seg.get("speaker", "UNKNOWN") for seg in segments))

    # Chuẩn bị font chữ tiếng Việt
    reg_font, bold_font = get_font_paths()
    font_name = "VietFont"

    pdf = MeetingPDF(font_name=font_name, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)

    if reg_font:
        pdf.add_font(font_name, "", reg_font)
        pdf.add_font(font_name, "B", bold_font if bold_font else reg_font)
    else:
        pdf.set_fallback_fonts(["Helvetica"])

    pdf.add_page()

    # 1. TIÊU ĐỀ CHÍNH
    pdf.set_font(font_name, "B", 16)
    pdf.set_text_color(24, 76, 120)
    pdf.cell(0, 10, meeting_title, align="C")
    pdf.ln(10)

    # Thông tin người tham gia
    pdf.set_font(font_name, "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Người tham gia: {', '.join(all_speakers)}", align="C")
    pdf.ln(6)

    # Đường kẻ ngang phân cách
    pdf.set_draw_color(24, 76, 120)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # 2. IN TỪNG PHÁT BIỂU THEO THỨ TỰ THỜI GIAN
    prev_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", "")
        end = seg.get("end", "")
        text_vn = seg.get("text_translated", "")
        text_orig = seg.get("text_original", seg.get("text", ""))

        # Khi đổi người nói → in tên speaker nổi bật
        if speaker != prev_speaker:
            pdf.ln(3)
            pdf.set_font(font_name, "B", 11)
            pdf.set_text_color(40, 116, 166)
            pdf.cell(0, 6, f"── {speaker} ──")
            pdf.ln(7)
            prev_speaker = speaker

        # Dòng thời gian + nội dung đã dịch
        if text_vn:
            cleaned = clean_line(text_vn)
            if cleaned:
                pdf.set_font(font_name, "B", 9)
                pdf.set_text_color(100, 100, 100)
                pdf.set_x(15)
                pdf.cell(35, 5, f"[{start} - {end}]")

                pdf.set_font(font_name, "", 10)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 5.5, cleaned)
                pdf.ln(1)

        # Nội dung gốc (in nhỏ, xám nhạt)
        if text_orig:
            orig_cleaned = clean_line(text_orig)
            if orig_cleaned:
                pdf.set_font(font_name, "", 8)
                pdf.set_text_color(150, 150, 150)
                pdf.set_x(50)
                pdf.multi_cell(0, 4.5, f"(Gốc: {orig_cleaned})")
                pdf.ln(1)

    # Xuất file
    pdf.output(output_pdf)
    print(f"\n ĐÃ XUẤT FILE PDF TRANSCRIPT THÀNH CÔNG: '{output_pdf}'")
    return output_pdf



