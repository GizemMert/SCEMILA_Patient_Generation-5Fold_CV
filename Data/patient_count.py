import os

def count_subfolders(main_folder):
    subfolders_count = {}

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            count = len([name for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))])
            subfolders_count[subfolder] = count

    return subfolders_count

def main():
    main_folder = '/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Data/data/data'
    subfolders_count = count_subfolders(main_folder)

    total_count = 0
    for subfolder, count in subfolders_count.items():
        print(f"{subfolder}: {count} subfolders")
        total_count += count

    print(f"Total number of subfolders: {total_count}")

if __name__ == "__main__":
    main()
