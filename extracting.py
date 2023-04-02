from zipfile import ZipFile
import os
import sys


def main():
    file_name = "titanic.zip"
    if len(sys.argv) == 2:
        file_name = sys.argv[-1]
    if file_name.endswith(".zip"):
        extract_zip_file(file_name)


def extract_zip_file(file_name):
    # parent_dir = os.path.join(os.path.dirname(__file__), "..")
    parent_dir = os.path.dirname(__file__)
    fn = os.path.join(parent_dir, file_name)
    if os.path.exists(fn):
        with ZipFile(fn, "r") as zObject:
            temp_dir_path = os.path.join(parent_dir, "temp")
            try:
                os.mkdir(temp_dir_path)
            except FileExistsError:
                print("Temp directory already exists!")
                temp_dir_files = os.listdir(temp_dir_path)
                for file in temp_dir_files:
                    os.remove(os.path.join(temp_dir_path, file))
                    print("file removed:", file)
            zObject.extractall(path=temp_dir_path)
            print("{} extracted to temp dir at {}".format(file_name, temp_dir_path))


if __name__ == '__main__':
    main()
