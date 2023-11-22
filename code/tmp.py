# import glob
# print (glob.glob("../video/ama_driver/*"))

# import os
# print(os.listdir('../video/ama_driver'))
import os
videoPath = "../video/progirl_driver/"

# for videoPath, dirs, videoName in os.walk(videoPath):
#     videoName = sorted(videoName)
# videoNum=len(videoName)

# print(videoName)
# print(videoNum)
# print('../data%stmpDiff.txt'%(videoPath[8:]))
import re

for videoPath, dirs, videoName in os.walk(videoPath):
    p = re.compile(r'\d+')
    print(sorted(videoName, key=lambda s: int(p.search(s).group())))

