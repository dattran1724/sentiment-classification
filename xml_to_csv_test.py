#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import csv
import sys
from pathlib import Path

def convert(xml_path: Path, csv_path: Path) -> None:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Giữ đúng cấu trúc cột bạn yêu cầu
        writer.writerow(["review_id", "sentence_id", "sentence_text", "target", "category", "polarity"]) 

        for review in root.findall('Review'):
            review_id = review.get('rid', '')
            sentences = review.find('sentences')
            if sentences is None:
                continue

            for sentence in sentences.findall('sentence'):
                sentence_id = sentence.get('id', '')
                text_el = sentence.find('text')
                sentence_text = text_el.text.strip() if text_el is not None and text_el.text else ''

                # Tìm thẻ Opinions
                opinions_parent = sentence.find('Opinions')
                
                # LOGIC CẬP NHẬT: Nếu không có thẻ Opinions (như file XML Phase A bạn gửi)
                if opinions_parent is None or len(opinions_parent.findall('Opinion')) == 0:
                    # Vẫn ghi nhận câu, để trống các trường nhãn
                    writer.writerow([review_id, sentence_id, sentence_text, '', '', ''])
                else:
                    # Nếu có Opinions, lặp qua từng Opinion để ghi dòng
                    for op in opinions_parent.findall('Opinion'):
                        target = op.get('target', 'NULL')
                        category = op.get('category', '')
                        polarity = op.get('polarity', '')
                        writer.writerow([review_id, sentence_id, sentence_text, target, category, polarity])

if __name__ == '__main__':
    # Đường dẫn
    input_dir = Path(r"D:\2025-2026\ChuyenDe\New_data\xml\test_gold")
    output_dir = Path(r"D:\2025-2026\ChuyenDe\New_data\csv\test_gold")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Lỗi: Thư mục nguồn không tồn tại: {input_dir}")
        sys.exit(1)

    # CẬP NHẬT: Tìm cả file .xml và .xml.gold
    xml_files = list(input_dir.glob("*.xml")) + list(input_dir.glob("*.xml.gold"))
    
    if not xml_files:
        print(f"Không tìm thấy file XML hoặc XML.gold nào.")
    else:
        print(f"Tìm thấy {len(xml_files)} file. Đang xử lý...")

    for xml_file in xml_files:
        # Giữ nguyên tên gốc, chỉ thay đuôi cuối cùng thành .csv
        csv_file = output_dir / (xml_file.name + ".csv")
        
        try:
            convert(xml_file, csv_file)
            print(f"Thành công: {xml_file.name} -> {csv_file.name}")
        except Exception as e:
            print(f"Lỗi tại file {xml_file.name}: {e}")

    print("\nHoàn tất quá trình chuyển đổi.")