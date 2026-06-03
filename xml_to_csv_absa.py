#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import csv
import sys
from pathlib import Path

def convert(xml_path: Path, csv_path: Path) -> None:
    # GIỮ NGUYÊN LOGIC XỬ LÝ
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "sentence_id", "sentence_text", "target", "category", "polarity"]) 

        for review in root.findall('Review'):
            review_id = review.get('rid', '')
            sentences = review.find('sentences')
            if sentences is None:
                continue

            for sentence in sentences.findall('sentence'):
                sentence_id = sentence.get('id', '')
                text_el = sentence.find('text')
                sentence_text = ''
                if text_el is not None and text_el.text is not None:
                    sentence_text = text_el.text.strip()

                opinions_parent = sentence.find('Opinions')
                if opinions_parent is None:
                    continue

                opinions = opinions_parent.findall('Opinion')
                if not opinions:
                    continue

                for op in opinions:
                    target = op.get('target', 'NULL')
                    category = op.get('category', '')
                    polarity = op.get('polarity', '')

                    writer.writerow([review_id, sentence_id, sentence_text, target, category, polarity])

if __name__ == '__main__':
    # Cấu hình đường dẫn cố định
    input_dir = Path(r"D:\2025-2026\ChuyenDe\New_data\xml\train")
    output_dir = Path(r"D:\2025-2026\ChuyenDe\New_data\csv\train")

    # Tạo thư mục đích nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Lỗi: Thư mục nguồn không tồn tại: {input_dir}")
        sys.exit(1)

    # Lặp qua tất cả file .xml trong thư mục nguồn
    xml_files = list(input_dir.glob("*.xml"))
    
    if not xml_files:
        print(f"Không tìm thấy file XML nào trong {input_dir}")
    else:
        print(f"Tìm thấy {len(xml_files)} file. Đang xử lý...")

    for xml_file in xml_files:
        # Tạo tên file csv tương ứng trong thư mục đích
        csv_file = output_dir / (xml_file.stem + ".csv")
        
        try:
            convert(xml_file, csv_file)
            print(f"Thành công: {xml_file.name} -> {csv_file.name}")
        except Exception as e:
            print(f"Lỗi tại file {xml_file.name}: {e}")

    print("\nHoàn tất quá trình chuyển đổi.")