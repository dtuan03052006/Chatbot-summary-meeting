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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Danh sách các vị trí có thể chứa font trong dự án
    font_candidates = [
        (os.path.join(current_dir, "fronts", "BeVietnamPro-Regular.ttf"),
         os.path.join(current_dir, "fronts", "BeVietnamPro-Bold.ttf")),
        (os.path.join(current_dir, "fonts", "BeVietnamPro-Regular.ttf"),
         os.path.join(current_dir, "fonts", "BeVietnamPro-Bold.ttf")),
        (os.path.join(current_dir, "BeVietnamPro-Regular.ttf"),
         os.path.join(current_dir, "BeVietnamPro-Bold.ttf")),
        (os.path.join(current_dir, "..", "fronts", "BeVietnamPro-Regular.ttf"),
         os.path.join(current_dir, "..", "fronts", "BeVietnamPro-Bold.ttf")),
    ]
    for reg, bld in font_candidates:
        if os.path.exists(reg) and os.path.exists(bld):
            return os.path.abspath(reg), os.path.abspath(bld)

    # 2. Kiểm tra các font hệ thống Linux / Kaggle có hỗ trợ UTF-8 tiếng Việt
    system_candidates = [
        ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for reg, bld in system_candidates:
        if os.path.exists(reg) and os.path.exists(bld):
            return reg, bld

    # 3. Tải font BeVietnamPro về thư mục /tmp/fonts
    target_dir = os.path.join(current_dir, "fonts")
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception:
        target_dir = "/tmp/fonts"
        os.makedirs(target_dir, exist_ok=True)

    target_reg = os.path.join(target_dir, "BeVietnamPro-Regular.ttf")
    target_bld = os.path.join(target_dir, "BeVietnamPro-Bold.ttf")

    if os.path.exists(target_reg) and os.path.exists(target_bld):
        return target_reg, target_bld

    print(" Đang tải font tiếng Việt BeVietnamPro từ Google Fonts...")
    try:
        req_reg = urllib.request.Request(URL_REGULAR, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_reg) as resp, open(target_reg, "wb") as f:
            f.write(resp.read())

        req_bld = urllib.request.Request(URL_BOLD, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_bld) as resp, open(target_bld, "wb") as f:
            f.write(resp.read())

        print(" Tải font thành công!")
        return target_reg, target_bld
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
    font_name = "VietFont" if reg_font else "Helvetica"

    pdf = MeetingPDF(font_name=font_name, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)

    # Đăng ký font UTF-8
    if reg_font:
        pdf.add_font(font_name, "", reg_font)
        pdf.add_font(font_name, "B", bold_font if bold_font else reg_font)

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
    font_name = "VietFont" if reg_font else "Helvetica"

    pdf = MeetingPDF(font_name=font_name, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)

    if reg_font:
        pdf.add_font(font_name, "", reg_font)
        pdf.add_font(font_name, "B", bold_font if bold_font else reg_font)

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
            if pdf.get_y() > 255:
                pdf.add_page()
            else:
                pdf.ln(4)
            pdf.set_font(font_name, "B", 11)
            pdf.set_text_color(24, 76, 120)  # Xanh dương đậm đồng bộ
            pdf.cell(0, 6, f"── {speaker} ──")
            pdf.ln(7)
            prev_speaker = speaker

        # Kiểm tra nếu gần cuối trang thì sang trang mới
        if pdf.get_y() > 268:
            pdf.add_page()

        curr_y = pdf.get_y()
        timestamp_str = f"[{start} - {end}]" if start and end else ""

        # 1. Cột trái: Mốc thời gian (X=15, rộng 46mm, không bao giờ bị đè)
        if timestamp_str:
            pdf.set_xy(15, curr_y)
            pdf.set_font(font_name, "B", 8.5)
            pdf.set_text_color(110, 110, 110)
            pdf.cell(46, 5, timestamp_str)

        # 2. Cột phải: Bản dịch tiếng Việt (X=63, rộng 132mm)
        cleaned_vn = clean_line(text_vn) if text_vn else ""
        if cleaned_vn:
            pdf.set_xy(63, curr_y)
            pdf.set_font(font_name, "", 9.5)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(132, 5.2, cleaned_vn)

        # 3. Cột phải: Bản gốc tiếng Anh (X=63, rộng 132mm)
        cleaned_orig = clean_line(text_orig) if text_orig else ""
        if cleaned_orig:
            pdf.set_x(63)
            pdf.set_font(font_name, "", 8)
            pdf.set_text_color(140, 140, 140)
            pdf.multi_cell(132, 4.2, f"(Gốc: {cleaned_orig})")

        pdf.ln(2.5)

    # Xuất file
    pdf.output(output_pdf)
    print(f"\n ĐÃ XUẤT FILE PDF TRANSCRIPT THÀNH CÔNG: '{output_pdf}'")
    return output_pdf



