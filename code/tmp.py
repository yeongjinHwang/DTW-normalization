# import glob
# print (glob.glob("../video/ama_driver/*"))

# import os
# print(os.listdir('../video/ama_driver'))
import os
videoPath = "../video/ama_driver/"

for videoPath, dirs, videoName in os.walk(videoPath):
    videoName = sorted(videoName)
videoNum=len(videoName)

print(videoName)
print(videoNum)

print('../data%stmpDiff.txt'%(videoPath[8:]))