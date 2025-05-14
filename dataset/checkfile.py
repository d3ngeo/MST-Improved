import os
import h5py

def check_mat_files(directory):
    """
    Check if .mat files in the given directory are corrupted.

    Args:
        directory (str): Path to the directory containing .mat files.
    """
    corrupted_files = []
    print(f"Checking .mat files in directory: {directory}")

    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            try:
                with h5py.File(file_path, 'r') as f:
                    # Attempt to read the file and access the 'cube' key (if applicable)
                    if 'cube' in f:
                        _ = f['cube'][:]
                print(f"File OK: {file_path}")
            except Exception as e:
                print(f"Corrupted file: {file_path}, Error: {e}")
                corrupted_files.append(file_path)

    print("\nSummary:")
    if corrupted_files:
        print("Corrupted files found:")
        for file in corrupted_files:
            print(file)
    else:
        print("No corrupted files found.")

# Directory to check
directory = '/media/ntu/volume1/home/s124md303_06/MST-plus-plus/dataset/Train_Spec/'

# Run the check
check_mat_files(directory)
