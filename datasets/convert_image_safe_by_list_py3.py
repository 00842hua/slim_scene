import os
from sys import argv
from PIL import Image
import logging


logging.basicConfig(level=logging.WARNING,
                format='%(asctime)s  %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

max_length = 1600

print(argv)

if len(argv) < 4:
    print("Need 3 params: image_list_file path_keyword_src path_keyword_dst [start_idx]")
    exit()

image_list_file = argv[1]
src = argv[2]
dst = argv[3]

start_idx = 0
if len(argv) >= 5:
    start_idx = int(argv[4])

print ('image_list_file: %s' % image_list_file)
print ('src: %s' % src)
print ('dst:: %s' % dst)
print ('start_idx: %s' % start_idx)

image_files = []
lines = open(image_list_file, 'r', encoding='UTF-8').read().splitlines()
print ("len(lines): %s" % len(lines))
image_files = lines
#image_files = [line.split()[1] for line in lines]
print ("len(image_files): %s" % len(image_files))
#print (image_files[0:10])

count = 0
total = len(image_files)
for one_image in image_files:
    count += 1
    if count % 500 == 0:
        logging.warning('Processed: %d / %d', count, total)
    if count <= start_idx:
        continue

    target_image = one_image.replace(src, dst)
    curr_target_dir = os.path.dirname(target_image)
    #print one_image, target_image, curr_target_dir

    if not os.path.isfile(one_image):
        print("===================Warning!!! %s is not a file" % one_image)
        continue

    try:
        image = Image.open(one_image)

        # 如果有旋转信息，直接旋转为对应方向
        try:
          for orientation in ExifTags.TAGS.keys() :
            if ExifTags.TAGS[orientation]=='Orientation' : break
          exif=dict(image._getexif().items())
          if  exif[orientation] == 3 :
            image=image.rotate(180, expand = True)
          elif exif[orientation] == 6 :
            image=image.rotate(270, expand = True)
          elif exif[orientation] == 8 :
            image=image.rotate(90, expand = True)
        except:
          pass

        #image = image.rotate(270, expand = True)
        mode = image.mode

        if mode != "RGB":
            image = image.convert("RGB")


        size = image.size
        if size[0] > max_length and size[1] > max_length:
            if size[0] < size[1]:
                new_size = (max_length, int(size[1]*max_length / size[0]))
            else:
                new_size = (int(size[0]*max_length / size[1]), max_length)
            image = image.resize(new_size)


        if not os.path.isdir(curr_target_dir):
            os.makedirs(curr_target_dir)
        image.save(target_image, format="JPEG")


    except Exception as e:
        print ("%s : %s" % (Exception, e))
        print ("---------- image: %s" % one_image)
