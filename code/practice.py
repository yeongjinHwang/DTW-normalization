import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math
from matplotlib.patches import ConnectionPatch
import scipy.spatial.distance as dist
import pandas as pd
from numpy.linalg import norm
from numpy import dot
import os
import re

videoPath = sys.argv[1]
if os.path.exists(videoPath)==False :
    print('path error')
    sys.exit()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image = image[0:960,0:540]
    image_height, image_width, _ = image.shape
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

####################################video select####################################

for videoPath, dirs, videoName in os.walk(videoPath):
    p = re.compile(r'\d+')
    videoName = sorted(videoName, key=lambda s: int(p.search(s).group()))

videoNum=len(videoName)

####################################video data road####################################
x, y = [], []  # (7, f, 33)
tmpx, tmpy = [], []

for i in range(videoNum):
    x.append([])
    y.append([])
    video = videoPath+videoName[i]
    cap = cv2.VideoCapture(video)
    width  = int(cap.get(3)) # float
    height = int(cap.get(4)) # float
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image = image[0:960,0:540]
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                for j in range(33):
                    if results.pose_landmarks==None :
                        tmpx.append(0.00001)
                        tmpy.append(0.00001)
                        continue
                    tmpx.append(round(results.pose_landmarks.landmark[j].x*width))
                    tmpy.append(round(results.pose_landmarks.landmark[j].y*height))
                x[i].append(tmpx)
                y[i].append(tmpy)
                tmpx=[]
                tmpy=[]

                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()
cv2.destroyAllWindows()
print("width:%d,height:%d"%(width,height))
print('data set complete')
#x[0] 첫번째영상 x[1] 두번째 영상, x[0][0] 첫 영상 첫프레임의 33개 데이터
#x[0][0][0] 첫 영상 첫 프레임 0번관절 x[0][0][1] 첫 영상 첫프레임 1번관절
#x[0][:][0] 첫 영상 모든 프레임 0번관절

videoEachFrame,angle = [], []
for i in range(videoNum) :
    angle.append([])
    videoEachFrame.append(len(x[i]))
    if i==0 :
        print('videoFrame : ',end=' ')
    print(videoEachFrame[i],end=' ')
    if i==videoNum-1 :
        print(' ')

####################################data(x,y)->data(angle)####################################
def angle_of_vectors(vec1,vec2) :
    a,b,c,d=vec1[0],vec1[1],vec2[0],vec2[1] 
    dotProduct = a*c + b*d 
    modOfVector1 = math.sqrt( a*a + b*b)*math.sqrt(c*c + d*d) 
    angle = dotProduct/modOfVector1
    if angle>1 :
        angle=1
    angleInDegree = math.degrees(math.acos(angle))
    return angleInDegree

# 12-11 어깨, 12-24 왼옆구리, 11-23 오른옆구리, 24-23 허리, 24-26 왼허벅 23-25 오른허벅
# 26-28 왼종아리, 25-27 오른종아리, 28-32 왼발등, 27-31 오른발등
# 12-14 왼팔, 11-13 오른팔, 14-16 왼전완, 13-15 오른전완
matchIndex=[[12,11],[12,24],[11,23],[24,23],[24,26],[23,25],[26,28],[25,27],[28,32],[27,31],
           [12,14],[11,13],[14,16],[13,15]]

#x[video][frame][angle]
#angle = [video][frame][angle1...angle14]

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

createDirectory('../data%s'%(videoPath[8:]))

for i in range(videoNum) :
    for frameNum in range(videoEachFrame[i]) :
        temp = []
        for idx in range(len(matchIndex)):
            ##print(idx) 지금 여기서 오류
            temp.append(angle_of_vectors([x[i][frameNum][matchIndex[idx][0]],y[i][frameNum][matchIndex[idx][0]]],
            [x[i][frameNum][matchIndex[idx][1]],y[i][frameNum][matchIndex[idx][1]]]))
        angle[i].append(temp)

    # angleDf = pd.DataFrame(angle[i])
    # angleDf.to_csv('../data%sangle%d.txt'%(videoPath[8:],i+1),index=False, sep='\t')
####################################DTW####################################

def dp(dist_mat):
    N, M = dist_mat.shape
    
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      
                cost_mat[i, j + 1],  
                cost_mat[i + 1, j]]  
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            i = i - 1
        elif tb_type == 2:
            j = j - 1
        path.append((i, j))

    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

dist_mat, N, M, path, cost_mat = [],[],[],[], []
for i in range(videoNum-1) :
    cost_mat.append([])
    dist_mat.append(np.zeros((videoEachFrame[0],videoEachFrame[i+1])))
    N.append(videoEachFrame[0])
    M.append(videoEachFrame[i+1])

####################################TOTAL average DTW####################################
# cost_mat[video][joint][videoframe][video+1frame]
for num in range(len(dist_mat)) : 
    for k in range(len(matchIndex)) :
        for i in range(N[num]) :
            for j in range(M[num]) :
                dist_mat[num][i, j] = abs(angle[0][i][k] - angle[num+1][j][k]) 
        cost_mat[num].append(dp(dist_mat[num])[1])

for num in range(len(cost_mat)):
    cost_mat[num] = np.asarray(cost_mat[num],dtype=object)

averageCostMat, path, least = [], [], []
for num in range(len(cost_mat)) :
    averageCostMat.append([])
    for joint in range(1,len(matchIndex)): 
        cost_mat[num][0][:,:]=cost_mat[num][0][:,:]+cost_mat[num][joint][:,:]
    cost_mat[num][0]=cost_mat[num][0]/len(matchIndex)
    averageCostMat[num] = cost_mat[num][0]

for num in range(1,videoNum-1) :
    averageCostMat[0][:,:] = averageCostMat[0][:,:] + averageCostMat[num][:,:]
averageCostMat[0] = averageCostMat[0] / (videoNum-1)

def reversePathFind(averMat) :
    i = averMat.shape[0]-1
    j = averMat.shape[1]-1
    if i==0 or j==0 :
        print('영상을 최소 2frame 이상 실행시켜주세요.')
        sys.exit()
    findPath = [(i,j)]
    while i>0 or j>0:
        a= averMat[i-1,j-1]
        b= averMat[i,j-1]
        c= averMat[i-1,j]
        if min(a,b,c)==a:
            i=i-1
            j=j-1
        elif min(a,b,c)==b:
            j=j-1
        elif min(a,b,c)==c:
            i=i-1
        findPath.append((i,j))
    return list(reversed(findPath))

path = []
path.append( reversePathFind(averageCostMat[0]) )
path = np.asarray(path,dtype=object) # path[6][baseVideoMatchingIndex][MatchingVideoIndex]
####################################Link index####################################
# LinkPath = np.full((videoEachFrame[0], videoNum),videoEachFrame[0]-1)

# for index in range(len(LinkPath)) :
#     LinkPath[index][0] = index

# for num in range(0,len(path)):
#     for linkIndex in range(len(LinkPath)) :
#         for pathIndex in range(min(len(LinkPath),len(path[num]))) :
#             if LinkPath[linkIndex][0] == path[num][pathIndex][0] :
#                 LinkPath[linkIndex][num+1] = path[num][pathIndex][1]
# LinkPath = np.zeros((videoEachFrame[0], videoNum))
# for frame in range(len(LinkPath)) :
#     LinkPath[frame][0] = frame

# for num in range(len(path)):
#     for frame in range(len(LinkPath)) :
#         for match in range(max(len(LinkPath),len(path[num]))) :
#                 if LinkPath[frame][0] == path[num][match][0] :
#                     LinkPath[frame][num+1] = path[num][match][1]

# linkDf = pd.DataFrame(LinkPath)
# linkDf.to_csv('../data%slinkPath.txt'%(videoPath[8:]),index=False,sep='\t')

####################################Average Value####################################
averValue = np.zeros((videoEachFrame[0],len(matchIndex)))
minDiffCnt = np.zeros((videoEachFrame[0],len(matchIndex)+1))
tmpDiff = np.zeros((len(matchIndex),videoNum))
numCnt = np.zeros((videoNum))
averVidIndex = 0

# for point in range(len(x[0][0])) :
#     for frame in range(len(x[0])) :
#         for num in range(len(x)) :
            
            ###########이쪽 현재 문제 ############
averValue[:][:] = averValue[:][:] + angle[0][int(path[0][:])][:]
for frame in range(len(videoNum)) :
    for joint in range(len(matchIndex)) :
            averValue[frame][joint] = averValue[frame][joint] + angle[num][int(path[1][])][joint]
    minDiffCnt[frame][len(matchIndex)]=np.inf
averValue = averValue/videoNum

for frame in range(len(LinkPath)) :
    for joint in range(len(matchIndex)) :
        for num in range(videoNum) :
            tmpDiff[joint][num] = abs(averValue[frame][joint] - angle[num][frame][joint])
        minDiffCnt[frame][joint] = np.argmin(tmpDiff[joint])
    for vidNum in range(videoNum) :
        numCnt[vidNum]=list(minDiffCnt[frame]).count(vidNum)
    minDiffCnt[frame][len(matchIndex)] = np.argmax(numCnt)

for vidNum in range(videoNum) :
    numCnt[vidNum] = list(minDiffCnt[:,len(matchIndex)]).count(vidNum)

averVidIndex = np.argmax(numCnt)
bestVid = videoName[averVidIndex]
print("best Video : ",bestVid)

# with open('../data%sbestVid.txt'%(videoPath[8:]), "w") as file:
#     file.write(bestVid)

# closeCntDf = pd.DataFrame(numCnt)
# closeCntDf.to_csv('../data%scloseCnt.txt'%(videoPath[8:]),index=False,sep='\t')
# tmpDiffDf = pd.DataFrame(tmpDiff)
# tmpDiffDf.to_csv('../data%stmpDiff.txt'%(videoPath[8:]),index=False,sep='\t')
# minDiffCntDf = pd.DataFrame(minDiffCnt)
# minDiffCntDf.to_csv('../data%sminDiffCnt.txt'%(videoPath[8:]),index=False,sep='\t')
# averValueDf = pd.DataFrame(averValue)
# averValueDf.to_csv('../data%saverValue.txt'%(videoPath[8:]),index=False,sep='\t')

##cv2.line(img,시작점,끝점,color(b,g,r),thickness,lineType,shift)
# 12-11 어깨, 12-24 왼옆구리, 11-23 오른옆구리, 24-23 허리, 24-26 왼허벅 23-25 오른허벅
# 26-28 왼종아리, 25-27 오른종아리, 28-32 왼발등, 27-31 오른발등
# 12-14 왼팔, 11-13 오른팔, 14-16 왼전완, 13-15 오른전완
# matchIndex=[[12,11],[12,24],[11,23],[24,23],[24,26],[23,25],[26,28],[25,27],[28,32],[27,31],
#            [12,14],[11,13],[14,16],[13,15]]

video, frame = videoPath + bestVid, 0
cap = cv2.VideoCapture(video)
width  = int(cap.get(3)) # float
height = int(cap.get(4)) # float
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        while frame<videoEachFrame[averVidIndex]:
            backgroundImage = np.zeros((960,540,3), np.uint8)  
            cap.set(cv2.CAP_PROP_POS_FRAMES,LinkPath[frame][averVidIndex])
            success, image = cap.read()
            if not success:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                backgroundImage,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            image = cv2.hconcat([image, backgroundImage])
            cv2.imshow('MediaPipe Pose', image)
            cv2.putText(image, "평균과 가장 근접한 skeleton/video", (50,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)
            frame+=1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
# video,cap = [], []
# image = []
# for num in range(videoNum) :
#     video.append(f"../video/pro1_iron{num+1}.mp4")
#     cap.append(cv2.VideoCapture(video[num]))
#     image.append(0)
# frame = 0
# while frame<len(LinkPath) :
#     for num in range(videoNum) :
#         cap[num].set(cv2.CAP_PROP_POS_FRAMES,LinkPath[frame][num])
#         image[num]=cap[num].read()[1]
#         image[num] = image[num][0:960, 0:540]
#     img = cv2.hconcat([image[0],image[1],image[2],image[3],image[4],image[5],image[6],image[7]])
#     cv2.imshow("Video",img)
#     frame+=1
#     if cv2.waitKey(5) & 0xFF == 'q' :
#         break
