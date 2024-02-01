import pandas as pd
import os, glob

class FilesList:
    def __init__(self, expression, class_index, base_path=None, list_file=None):
        """
        Initialize the FilesList instance by either loading from a file or generating from a base path.
        Provides feedback on the number of files and classes processed.
        """
        self.files_df = pd.DataFrame(columns=['class', 'file'])
        self.index = 0  # Iterator index

        if list_file and os.path.exists(list_file):
            print("Loading list from file...")
            self.load_from_file(list_file)
            num_files = len(self.files_df)
            num_classes = self.files_df['class'].nunique()
            print(f"List loaded successfully from {list_file}.")
            print(f"Total files loaded: {num_files}. Unique classes: {num_classes}.")
        elif base_path:
            if self._test_dataset_structure(base_path, expression, class_index):
                print("Generating list from base path...")
                self.generate_from_base_path(base_path, expression, class_index)
                num_files = len(self.files_df)
                num_classes = self.files_df['class'].nunique()
                print("List generated successfully.")
                print(f"Total files found: {num_files}. Unique classes: {num_classes}.")
            else:
                print("Dataset structure validation failed. The dataset does not match the expected pattern.")
        else:
            print("Warning: No base_path or list_file provided. The DataFrame is empty.")

    def generate_from_base_path(self, base_path, expression, class_index):
        """
        Generates a list of files from a specified base path and updates the class's DataFrame with this list.

        If class_index points to the filename, the file extension is removed and used as the class.
        """
        temp_list = []  # Temporary list to store dictionaries

        files_glob = glob.glob(os.path.join(base_path, expression))
        files_glob.sort()

        for f in files_glob:
            f = f.replace('\\', '/')
            file_rel_path = f.replace(base_path, '').lstrip('/')
            path_parts = file_rel_path.split('/')

            # Determine the class based on class_index
            if class_index >= len(path_parts):
                raise ValueError("class_index is out of range for the path depth.")

            cls = path_parts[class_index]
            if class_index == len(path_parts) - 1:  # If class_index points to the file name
                cls, _ = os.path.splitext(cls)  # Remove file extension to use filename as class

            temp_list.append({'class': cls, 'file': file_rel_path})

        # Convert the temporary list to a DataFrame in one operation
        self.files_df = pd.DataFrame(temp_list)

    def load_from_file(self, list_file):
        self.files_df = pd.read_csv(list_file, sep=';', names=['class', 'file'], skiprows=1)

    def save_to_file(self, output_file):
        self.files_df.to_csv(output_file, sep=';', index=False, header=['class', 'file'])

    def __iter__(self):
        return self.files_df.iterrows()

    def __next__(self):
        if self.index < len(self.files_df):
            result = self.files_df.iloc[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def _test_dataset_structure(self, base_path, expression, class_index=0):
        """
        Tests if the dataset in the base_path is organized according to the expected directory structure.

        This is an internal method used to validate the dataset structure before processing.

        Parameters:
        - base_path (str): The root directory to start searching for files.
        - expression (str): A glob pattern used to match files within the base_path.
        - class_index (int): The directory level expected to represent the class.

        Returns:
        - bool: True if the dataset structure matches the expectation, False otherwise.
        """
        sample_files = glob.glob(os.path.join(base_path, expression), recursive=True)[:10]  # Check a sample of 10 files

        if not sample_files:
            print("No files found matching the pattern. Please check the base_path and expression.")
            return False

        for file_path in sample_files:
            relative_path = os.path.relpath(file_path, base_path)
            path_parts = relative_path.replace('\\', '/').split('/')
            if len(path_parts) - 1 < class_index:  # Subtract 1 because the file itself is included in path_parts
                print(f"Directory depth does not match class_index for file: {file_path}")
                return False

        print("Dataset structure matches the expected pattern.")
        return True
