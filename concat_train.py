import pandas as pd
import os

# Đường dẫn đến thư mục chứa các file CSV vừa chuyển đổi
csv_dir = r"D:\2025-2026\ChuyenDe\New_data\csv\train\2016"
# Đường dẫn file đầu ra
output_file = os.path.join(csv_dir, "train_V1.csv")

# Danh sách lưu các dataframe
all_df = []

# Lặp qua các file trong thư mục
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv") and filename != "test_gold.csv":
        file_path = os.path.join(csv_dir, filename)
        # Đọc file và thêm vào danh sách
        df = pd.read_csv(file_path)
        all_df.append(df)

if all_df:
    # Nối tất cả các dataframe lại thành một
    combined_df = pd.concat(all_df, ignore_index=True)
    
    # Lưu thành file train.csv
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Đã nối thành công {len(all_df)} file vào: {output_file}")
else:
    print("Không tìm thấy file CSV nào để nối.")