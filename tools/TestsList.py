import pandas as pd
import numpy as np
from tqdm import tqdm
import os, csv, torch
from tools.FilesList import FilesList

class TestsList:
    def __init__(self, files_list=None, testslist_file=None, files_list_file=None, min_files=None, max_files=None,
                 n_ref=1, positive_reps=4, negative_reps=4, rs=42):
        """
        Initialize the TestsList instance by either using an existing FilesList instance,
        loading from a test list file, or loading from a files list file.

        Parameters:
        - files_list (FilesList): An instance of FilesList from which to generate the test list.
        - test_list_file (str): Path to a file containing a pre-defined test list.
        - files_list_file (str): Path to a file containing a files list, from which a FilesList instance will be created.
        - min_files (int): Minimum number of files per class for inclusion in the test list.
        - max_files (int): Maximum number of files per class for inclusion in the test list.
        - n_ref (int): Number of reference files to include in each test.
        - positive_reps (int): Number of positive test cases to generate per probe file.
        - negative_reps (int): Number of negative test cases to generate per probe file.
        - rs (int): Random state seed for reproducibility.
        """
        self.min_files = min_files
        self.max_files = max_files
        self.n_ref = n_ref
        self.positive_reps = positive_reps
        self.negative_reps = negative_reps
        self.rs = rs  # Random state for reproducibility

        # Initialize the test DataFrame with columns for test_type, probe_file, and dynamically added reference_files
        self.test_df = pd.DataFrame(columns=['test_type', 'probe_file'] + [f'ref_{i + 1}' for i in range(n_ref)])

        if files_list:
            self.generate_tests_list(files_list)
        elif testslist_file and os.path.exists(testslist_file):
            self._load_testslist_from_file(testslist_file)
        elif files_list_file and os.path.exists(files_list_file):
            # Instantiate a FilesList object from the files list file
            files_list = FilesList(list_file=files_list_file)
            self.generate_from_files_list(files_list)
        else:
            print("Warning: No valid input provided. The DataFrame is empty.")

    def _filter_classes_by_file_count(self, df, class_column, file_column, min_files, max_files=None):
        """
        Filters classes by the number of files, keeping only those within the specified range.
        Also tracks classes excluded by not reaching the minimum files per class and those not fully explored.
        """
        if max_files is None:
            max_files = df[class_column].value_counts().max()  # Use the maximum count if max_files is not specified

        # Determine class counts
        class_counts = df[class_column].value_counts()

        # Identify excluded and not fully explored classes
        self.excluded_classes = class_counts[class_counts < min_files].index.tolist()
        self.not_fully_explored_classes = class_counts[class_counts > max_files].index.tolist()

        # Filter classes by min_files
        valid_classes = class_counts[class_counts >= min_files].index
        filtered_df = df[df[class_column].isin(valid_classes)]

        # Randomly select max_files from each class if necessary
        def sample_max_files(group):
            return group.sample(n=min(len(group), max_files), random_state=self.rs)

        result_df = filtered_df.groupby(class_column).apply(sample_max_files).reset_index(drop=True)

        # Report the number of excluded and not fully explored classes
        print(f"Excluded classes (less than {min_files} files): {len(self.excluded_classes)}")
        print(f"Classes not fully explored (more than {max_files} files): {len(self.not_fully_explored_classes)}")

        return result_df

    def __repr__(self):
        return (f"TestsList(n_ref={self.n_ref}, positive_reps={self.positive_reps}, "
                f"negative_reps={self.negative_reps}, min_files={self.min_files}, "
                f"max_files={self.max_files}, num_tests={self.n_tests}, "
                f"n_positives={self.n_positives}, n_negatives={self.n_negatives})")

    def _filter_classes_by_file_count_from_files_df(self, files_df):
        # Group by class and filter based on min_files
        grouped = files_df.groupby('class')
        # Ensure self.min_files is an integer or None; if None, do not filter
        if self.min_files is not None:
            filtered_groups = grouped.filter(lambda x: len(x) >= self.min_files)
        else:
            filtered_groups = files_df  # If min_files is None, skip filtering

        # Define a function to randomly select max_files from each class, if necessary
        def sample_max_files(group):
            # If self.max_files is not None, sample up to self.max_files from the group
            if self.max_files is not None:
                return group.sample(n=min(len(group), self.max_files), random_state=self.rs)
            # If self.max_files is None, return the group as is
            return group

        # Apply sampling based on max_files and return the result
        return filtered_groups.groupby('class').apply(sample_max_files).reset_index(drop=True)

    def generate_tests_list(self, files_list):
        # Load dataset
        files_df = files_list.files_df

        # Filter and prepare the dataset
        if self.min_files:
            files_df = self._filter_classes_by_file_count_from_files_df(files_df)

        valid_classes = list(files_df.groupby('class').filter(lambda x: len(x) > self.n_ref)['class'].unique())
        files_df = files_df[files_df['class'].isin(valid_classes)].reset_index(drop=True)

        print('Valid classes:', len(valid_classes))

        # Setup the device for PyTorch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create class to indices mapping
        class_to_indices = {cls: torch.tensor(files_df[files_df['class'] == cls].index.values, device=device) for cls in valid_classes}

        # Create a dictionary for other classes
        other_classes_dict = {cls: [ocls for ocls in valid_classes if ocls != cls] for cls in valid_classes}

        # Initialize an empty list to collect test cases
        test_cases = []

        def generate_tests(i, existing_tests):
            tests = []

            probe_file = files_df.loc[i, 'file']
            probe_class = files_df.loc[i, 'class']

            # Filter out the probe index and create a mask for remaining indices
            remaining_indices = class_to_indices[probe_class] != i
            filtered_indices = class_to_indices[probe_class][remaining_indices]

            # Generate positive tests
            for _ in range(self.positive_reps):
                if len(filtered_indices) > self.n_ref:
                    ref_indices = filtered_indices[torch.randperm(len(filtered_indices), device=device)[:self.n_ref]]
                else:
                    ref_indices = filtered_indices
                ref_files = [files_df.loc[idx.item(), 'file'] for idx in ref_indices.cpu()]
                test_case = {'test_type': 1, 'probe_file': probe_file, **{f'ref_{j + 1}': ref_files[j] for j in range(len(ref_files))}}
                test_string = f"1;{probe_file};{';'.join(ref_files)}"
                if test_string not in existing_tests:
                    tests.append(test_case)
                    existing_tests.add(test_string)

            # Generate negative tests
            for _ in range(self.negative_reps):
                other_classes = np.random.choice(other_classes_dict[probe_class], self.n_ref, replace=False)
                ref_files = [files_df.loc[class_to_indices[cls][torch.randint(0, len(class_to_indices[cls]), (1,), device=device)].item(), 'file'] for cls in other_classes]
                test_case = {'test_type': 0, 'probe_file': probe_file, **{f'ref_{j + 1}': ref_files[j] for j in range(len(ref_files))}}
                test_string = f"0;{probe_file};{';'.join(ref_files)}"
                if test_string not in existing_tests:
                    tests.append(test_case)
                    existing_tests.add(test_string)

            return tests

        # Generate all tests
        existing_tests = set()
        for i in tqdm(range(len(files_df))):
            test_cases.extend(generate_tests(i, existing_tests))

        # Populate self.test_df with the generated test cases
        self.test_df = pd.DataFrame(test_cases, columns=['test_type', 'probe_file'] + [f'ref_{i + 1}' for i in range(self.n_ref)])

        # Count positive and negative tests
        self.n_positives = len(self.test_df[self.test_df['test_type'] == 1])
        self.n_negatives = len(self.test_df[self.test_df['test_type'] == 0])
        self.n_tests = len(self.test_df)


    def _load_testslist_from_file(self, testslist_file):
        """
        Main method to load the test list from a file after passing all validations.
        """
#        try:
        # Step 1: Detect file format (separator and header)
        dialect, has_header = self._detect_file_format(testslist_file)
        if dialect is None or has_header is None:
            raise ValueError("Failed to detect file format.")

        # Step 2: Validate if the columns match the expected structure
        if not self._validate_column_structure(testslist_file, dialect, has_header):
            raise ValueError("Column structure does not match expected format.")

        # Step 3: Load file into DataFrame
        # Load file into a temporary DataFrame with the correct structure
        temp_df = None
        if has_header:
            # Load with headers, assuming validation has confirmed the structure
            temp_df = pd.read_csv(testslist_file, dialect=dialect)
        else:
            # Load without headers and manually assign column names
            column_names = ['test_type', 'probe_file'] + [f'ref_{i + 1}' for i in range(self.n_ref)]
            temp_df = pd.read_csv(testslist_file, dialect=dialect, header=None, names=column_names)

        # Step 4: Validate the content of the columns
        is_valid, corrected_df = self._validate_column_content(temp_df)
        if not is_valid:
            raise ValueError("Column content validation failed.")

        # Assuming validation passed, and corrected_df contains the normalized paths
        self.test_df = corrected_df

        # Count positive and negative tests
        self.n_positives = len(self.test_df[self.test_df['test_type'] == 1])
        self.n_negatives = len(self.test_df[self.test_df['test_type'] == 0])
        self.n_tests = len(self.test_df)

        #except Exception as e:
        #    print(f"Error loading test list: {e}")
        #    # Optionally, log the error or handle it further here
        #    return None

    # Placeholder for _detect_file_format method
    def _detect_file_format(self, test_list_file):
        """
        Detects the CSV file's dialect (including the separator) and checks for a header row.

        Parameters:
        - test_list_file: Path to the CSV file to be analyzed.

        Returns:
        - dialect: The detected CSV dialect.
        - has_header: Boolean indicating whether the file has a header row.
        """
        try:
            with open(test_list_file, 'r', newline='') as file:
                # Sniff the first 1024 bytes to detect the dialect
                dialect = csv.Sniffer().sniff(file.read(1024))
                file.seek(0)  # Reset the file read position

                # Use the Sniffer to check if the file has a header row
                has_header = csv.Sniffer().has_header(file.read(1024))
                file.seek(0)  # Reset the file read position again

                return dialect, has_header
        except csv.Error as e:
            print(f"CSV format detection error: {e}")
            return None, None

    # Placeholder for _validate_column_structure method
    def _validate_column_structure(self, testslist_file, dialect, has_header):
        """
        Validates the structure of the columns in the CSV file.

        Parameters:
        - test_list_file: Path to the CSV file to be validated.
        - dialect: The detected CSV dialect from the _detect_file_format method.
        - has_header: Boolean indicating whether the file has a header row.
        - n_ref: Number of reference file columns expected.

        Returns:
        - Boolean indicating whether the column structure is valid.
        """
        try:
            # Load the file with the detected settings to check its structure
            df = pd.read_csv(testslist_file, dialect=dialect, header=0 if has_header else None)

            # Define the expected columns
            expected_columns = ['test_type', 'probe_file'] + [f'ref_{i + 1}' for i in range(self.n_ref)]

            if has_header:
                # When there is a header, check if all expected columns are present
                missing_columns = set(expected_columns) - set(df.columns)
                if missing_columns:
                    print(f"Missing required columns: {missing_columns}")
                    return False

                # Additional checks for column structure could be added here if necessary
            else:
                # If there is no header, ensure there are at least 3 columns (test_type, probe_file, at least one ref_file)
                if df.shape[1] < 3:
                    print("The file must contain at least 3 columns.")
                    return False

                # Assign default column names assuming the first two are test_type and probe_file, followed by ref_files
                df.columns = expected_columns[:df.shape[1]]

            return True  # The column structure is valid
        except Exception as e:
            print(f"Error validating column structure: {e}")
            return False

    def _validate_column_content(self, df):
        # Normalize and validate 'test_type' column
        if not self._validate_test_type_column(df['test_type']):
            return False, df

        # Normalize and validate file path columns
        file_path_columns = [col for col in df.columns if col != 'test_type']
        for col in file_path_columns:
            df[col] = df[col].apply(
                lambda path: self._validate_and_normalize_file_path(path) if isinstance(path, str) else path)
            if df[col].isnull().any():
                print(f"Invalid file path detected in column '{col}'.")
                return False, df

        return True, df

    def _validate_test_type_column(self, series):
        """
        Validates and converts 'test_type' values to 0 or 1. Handles both numeric and string representations,
        including case-insensitive string comparisons.
        """
        # Update the conversion to handle both string and numeric representations
        conversion_dict = {
            'true': 1, 'false': 0,
            'positive': 1, 'negative': 0
        }
        # Convert string representations to lowercase and apply the conversion
        # Numeric values are directly compared
        try:
            converted = series.apply(lambda x: conversion_dict.get(str(x).lower(), x))
            # Ensure all values are either 0 or 1 after conversion
            if converted.isin([0, 1]).all():
                return True
            else:
                return False
        except Exception as e:
            print(f"Error during test_type conversion: {e}")
            return False

    def _validate_and_normalize_file_path(self, path):
        """
        Normalizes a file path to Linux/Unix standard and validates it.
        Returns the normalized path if valid, otherwise returns None.
        """
        # Normalize path
        normalized_path = path.replace('\\', '/')
        # Validate normalized path
        if isinstance(normalized_path, str) and '/' in normalized_path and '.' in normalized_path.split('/')[-1]:
            return normalized_path  # Return the valid, normalized path
        else:
            return None  # Indicate invalid path

    def load_from_test_list_file(self, test_list_file):
        """
        Loads the test list from a specified test list file, handling files with or without headers,
        supporting multiple separators, and ensuring content consistency.
        """
        # Attempt to detect the separator and validate column consistency
        sep, consistent, has_header = self._detect_separator_and_consistency(test_list_file)

        if not consistent:
            raise ValueError("Inconsistent number of columns detected across lines. Please check the file format.")

        # Load the dataset with the detected settings
        df = pd.read_csv(test_list_file, sep=sep, header=0 if has_header else None, quotechar='"')

        print(df)

        # If no header, assign default column names
        if not has_header:
            df.columns = ['test_type'] + [f'file{i}' for i in range(1, len(df.columns))]

        # Convert test_type to 0 or 1 based on content
        df['test_type'] = df['test_type'].apply(lambda x: 1 if str(x).strip() in ['1', 'positive'] else 0)

        # Store the loaded and processed DataFrame in self.test_df
        self.test_df = df

        # Count positive and negative tests
        self.n_positives = len(self.test_df[self.test_df['test_type'] == 1])
        self.n_negatives = len(self.test_df[self.test_df['test_type'] == 0])
        self.n_tests = len(self.test_df)

        print(f"Loaded {len(self.test_df)} tests from '{test_list_file}' with separator '{sep}'.")

    def _detect_separator_and_consistency(self, test_list_file):
        """
        Detects the file's separator and checks if all lines have the same number of columns.
        Also attempts to determine if the content is enclosed by double quotes and if there's a header.
        """
        with open(test_list_file, 'r') as file:
            lines = file.readlines()

        separators = [';', ',', ' ']
        for sep in separators:
            first_line_cols = len(lines[0].split(sep))
            quoted = '"' in lines[0]

            if quoted:
                # Adjust for quoted separator
                first_line_cols = len([col for col in lines[0].split(sep) if '"' in col])

            consistent = all(len(line.split(sep)) == first_line_cols or '"' in line and len([col for col in line.split(sep) if '"' in col]) == first_line_cols for line in lines[1:])

            if consistent:
                # Check if the first line is likely to be a header
                has_header = not lines[0].replace(sep, '').strip().replace('.', '').replace('"', '').isdigit()
                return sep, consistent, has_header

        # If no separator found that makes the file consistent, return the first separator with a consistency flag set to False
        return separators[0], False, False

    def save_test_list_to_file(self, output_file):
        """
        Saves the test list to a specified file as a semicolon-separated CSV without a header.

        Parameters:
        - file_name (str): The name of the file where the test list will be saved.
        """
        if not self.test_df.empty:
            self.test_df.to_csv(output_file, sep=';', index=False, header=False)
            print(f"Test list successfully saved to '{output_file}'.")
        else:
            print("Warning: The test list DataFrame is empty. No file was saved.")
