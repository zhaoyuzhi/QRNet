import os

option_dir = "./options"
input_dir = "./models"
val_log_dir = "./val_logs"
val_output_dir = "./val_results"

def get_files(path):
    # read a folder, return the image name
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
    
def get_subfolders(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            fullpath = os.path.join(root, filespath)
            subfolder = fullpath.split('\\')[1]
            if subfolder + '.txt' not in ret:
                ret.append(subfolder + '.txt')
    return ret

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

# all the options
alloplist = get_jpgs(option_dir)
print('There should be %d options' % len(alloplist))

# all the prospective folder names, i.e., methods
subfolderlist = get_subfolders(input_dir)
print('There are %d subfolders' % len(subfolderlist))

# all the saved logs
alltxtlist = get_jpgs(val_log_dir)
selectedtxtlist = []

# renewed logs
for fname in alltxtlist:
    #print(fname)
    if fname in subfolderlist:
        selectedtxtlist.append(fname)

# print PSNR / SSIM
for fname in selectedtxtlist:
    alllines = text_readlines(os.path.join(val_log_dir, fname))
    val_result_folder_path = os.path.join(val_output_dir, fname.split('.txt')[0])
    try:
        len(get_jpgs(val_result_folder_path)) == 30
    except AssertionError:
        print('%s results not downloaded' % (val_result_folder_path))
    else:
        for i in range(len(alllines)):
            if 'The average of' in alllines[i]:
                str_psnr = alllines[i].split('PSNR')[1][2:7]
                str_ssim = alllines[i].split('SSIM')[1][2:8]
                print('%s: PSNR %s, SSIM %s' % (fname, str_psnr, str_ssim))
            