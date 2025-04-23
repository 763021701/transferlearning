import os
import argparse

def calculate_average_transfer_result(directory):
    total = 0
    count = 0

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.log'):
            file_path = os.path.join(directory, filename)

            # 打开并读取文件的最后一行
            with open(file_path, 'r') as file:
                lines = file.readlines()
                last_line = lines[-1].strip()

                # 提取 Transfer result 的数值部分
                if last_line.startswith('Transfer result:'):
                    try:
                        transfer_result = float(last_line.split(': ')[1])
                        total += transfer_result
                        count += 1
                    except ValueError:
                        print(f"Error converting to float in file: {filename}")

    # 计算平均值
    if count > 0:
        average = total / count
        return average
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the average Transfer result from .log files.')
    parser.add_argument('directory', type=str, help='The directory containing the .log files')
    args = parser.parse_args()

    directory_path = args.directory
    average_result = calculate_average_transfer_result(directory_path)

    if average_result is not None:
        print(f"The average Transfer result is: {average_result:.4f}")
    else:
        print("No valid .log files found or no Transfer results could be extracted.")




