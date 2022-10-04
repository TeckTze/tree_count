import os
import glob
import cv2

def main():

    folder = r'/home/team-ai/TD_API/output'
    png_files = glob.glob(os.path.join(folder, '*.png'))
    aspect_ratio_set = set()

    for png_file in png_files:

        img = cv2.imread(png_file)
        h,w = img.shape[:2]

        if (h,w) not in aspect_ratio_set:
            print((h,w))
            aspect_ratio_set.add((h,w))

    print(aspect_ratio_set)


if __name__ == '__main__':
    main()
