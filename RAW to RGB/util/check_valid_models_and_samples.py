import os

sample_dir = "./samples"
model_dir = "./models"
val_result_dir = "./val_results"

def get_files(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_subfolders(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            fullpath = os.path.join(root, filespath)
            subfolder = fullpath.split('\\')[1]
            if subfolder not in ret:# and 'quadbayer_ls' not in subfolder:
                ret.append(subfolder)
    return ret

subfolderlist = get_subfolders(model_dir)
print('There are %d subfolders for %s' % (len(subfolderlist), model_dir))

# 判断：下载了训练中途保存的sample和训练好的model之后，没有做test的模型
for fname in subfolderlist:
    determine = True
    ret = get_files(os.path.join(sample_dir, fname))
    if len(ret) != 600:
        determine = False
    ret = get_files(os.path.join(model_dir, fname))
    if len(ret) != 2:
        determine = False
    ret = get_files(os.path.join(val_result_dir, fname))
    if len(ret) == 30:
        determine = False
    if determine and 'quadbayer_ls' in fname:
        print(fname)
