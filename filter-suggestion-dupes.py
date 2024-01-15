import sys
import os

def parse_and_check_file_existence(input_data, directory_path):
    formatted_output = []
    files_in_directory = list(os.listdir(directory_path))

    for i in range(0, len(input_data), 2):
        if i + 1 < len(input_data) and input_data[i + 1].startswith('curl'):
            name = input_data[i]
            command = input_data[i + 1]
            file_check_string = f"{name.lower()}.safetensors"
            if file_check_string not in files_in_directory:
                formatted_output.append(f"{name}\n{command}")

    return "\n\n".join(formatted_output)

def main():
    directory_path = "loras"  # Replace with the actual path to the 'loras' directory
    input_data = sys.stdin.read().splitlines()
    output = parse_and_check_file_existence(input_data, directory_path)
    print(output)

if __name__ == "__main__":
    main()
