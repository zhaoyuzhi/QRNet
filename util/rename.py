import os

input_dir = "./options"
output_dir = "./options"

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

imglist = get_files(input_dir)

for fname in imglist:
    print(fname)
    path1 = fname
    path2 = fname.replace(".sh", ".yaml")
    os.renames(path1, path2)
