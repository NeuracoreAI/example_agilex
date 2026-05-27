import pathlib


def combine_python_files(directory_path, output_filename):
    """
    Recursively finds all .py files in a directory and combines them into a single text file.

    Args:
        directory_path (str): The path to the root directory to search.
        output_filename (str): The name/path of the output text file.
    """
    # Create a Path object for the target directory
    source_dir = pathlib.Path(directory_path)

    # Ensure the directory exists
    if not source_dir.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    # Open the output file in write mode
    with open(output_filename, "w", encoding="utf-8") as outfile:
        # rglob('*.py') recursively searches for all files ending in .py
        py_files = list(source_dir.rglob("*.py"))

        if not py_files:
            print(f"No Python files found in '{directory_path}'.")
            return

        print(f"Found {len(py_files)} Python files. Combining...")

        for file_path in py_files:
            # Write a clear separator and the file path as a header
            outfile.write(f"\n{'='*60}\n")
            outfile.write(f"FILE: {file_path}\n")
            outfile.write(f"{'='*60}\n\n")

            # Read the python file and append its contents
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
            except Exception as e:
                error_msg = f"Error reading file {file_path}: {e}\n"
                print(error_msg)
                outfile.write(error_msg)

    print(f"Success! All files have been combined into '{output_filename}'.")


# --- Example Usage ---
if __name__ == "__main__":
    # Replace these variables with your actual paths
    TARGET_DIRECTORY = "./"  # "./" means current directory
    OUTPUT_FILE = "combined_code.txt"

    combine_python_files(TARGET_DIRECTORY, OUTPUT_FILE)
