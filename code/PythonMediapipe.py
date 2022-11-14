import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import sys

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


####################################video data road####################################
x, y = [[], [], [], [], [], [], []], [[], [], [], [], [], [], []]  # (7, f, 33)
tmpx, tmpy = [], []
videoNum=7

for i in range(1,videoNum+1):
    video = "../video/front%d.mp4" % (i) 
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
                    tmpx.append(round(results.pose_landmarks.landmark[j].x*width))
                    tmpy.append(round(results.pose_landmarks.landmark[j].y*height))
                x[i-1].append(tmpx)
                y[i-1].append(tmpy)
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

videoEachFrame=[len(x[0]),len(x[1]),len(x[2]),len(x[3]),len(x[4]),len(x[5]),len(x[6])]

for i in range(videoNum) :
    if i==0 :
        print('videoFrame : ',end=' ')
    print(videoEachFrame[i],end=' ')
    if i==6 :
        print(' ')

####################################data(x,y)->data(angle)####################################
import math

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
angle = [[], [], [], [], [], [], []]

for i in range(videoNum) :
    for frameNum in range(videoEachFrame[i]) :
        temp = []
        for idx in range(len(matchIndex)):
            temp.append(angle_of_vectors([x[i][frameNum][matchIndex[idx][0]],y[i][frameNum][matchIndex[idx][0]]],
            [x[i][frameNum][matchIndex[idx][1]],y[i][frameNum][matchIndex[idx][1]]]))
        angle[i].append(temp)
####################################DTW####################################

from matplotlib.patches import ConnectionPatch
import scipy.spatial.distance as dist

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

# x = np.asarray(angle[0])                                 
# y = np.asarray(angle[1])                                   
# N = videoEachFrame[0]
# M = videoEachFrame[1]
dist_mat, N, M, path= [],[],[],[]
for i in range(videoNum-1) :
    dist_mat.append(np.zeros((videoEachFrame[i],videoEachFrame[i+1])))
    N.append(videoEachFrame[i])
    M.append(videoEachFrame[i+1])
# print(N, M)
# print(dist_mat[0])
####################################TOTAL average DTW####################################
cost_mat = [[],[],[],[],[],[]] # cost_mat[video][joint][videoframe][video+1frame]
for num in range(len(dist_mat)) : 
    for k in range(len(matchIndex)) :
        for i in range(N[num]) :
            for j in range(M[num]) :
                dist_mat[num][i, j] = abs(angle[num][i][k] - angle[num+1][j][k]) #0 video shoulder vs 1 video shoulder
        cost_mat[num].append(dp(dist_mat[num])[1])
        
cost_mat = np.asarray(cost_mat,dtype=object)
averageCostMat, path, least = [], [], []

for num in range(len(cost_mat)) :
    averageCostMat.append([])
    for joint in range(1,len(matchIndex)): 
        cost_mat[num][0][:,:]=cost_mat[num][0][:,:]+cost_mat[num][joint][:,:]
    cost_mat[num][0]=cost_mat[num][0]/len(matchIndex)
    averageCostMat[num] = cost_mat[num][0]

def reversePathFind(averMat) :
    i = averMat.shape[0]-1
    j = averMat.shape[1]-1
    print(i,j)
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
for num in range(len(averageCostMat)) :
    path.append( reversePathFind(averageCostMat[num]) )
    path[num] = np.asarray(path[num],dtype=object)
path = np.asarray(path,dtype=object)

# print(len(cost_mat),len(cost_mat[0]),len(cost_mat[0][0]),len(cost_mat[0][0][0]))     
# cost_mat = np.ndarray(cost_mat)
# cost_mat=float(cost_mat)
# print(len(cost_mat),len(cost_mat[0]),len(cost_mat[0][0]))
# totalCostMat=[[],[],[],[],[],[]]

# totalCostMap=[[],[],[],[],[],[]]
# for i in range(len(matchIndex)):
#     totalCostMap[]=totalCostMap+totalcost[i][:,:]
# totalCostMap = totalCostMap/len(matchIndex)
# totalCostMap = np.asarray(totalCostMap)
# path=[(totalCostMap.shape[0]-1,totalCostMap.shape[1]-1)]
# i=totalCostMap.shape[0]-1
# j=totalCostMap.shape[1]-1

# while i>0 or j>0:
#     a= totalCostMap[i-1,j-1]
#     b= totalCostMap[i,j-1]
#     c= totalCostMap[i-1,j]
#     if min(a,b,c)==a:
#         i=i-1
#         j=j-1
#     elif min(a,b,c)==b:
#         j=j-1
#     elif min(a,b,c)==c:
#         i=i-1
#     path.append((i,j))
# path.reverse()
# print('dtw len : ',len(path))

####################################Link index####################################

LinkPath = []
for num in range(1,len(path)) :
    print(num-1)
    for index in path[num-1][:,num] :
        for index2 in path[num][:,0] :



####################################Video execute####################################
video,cap = [], []

for num in range(videoNum) :
    video.append(f"../video/front{num+1}.mp4")
    cap.append(cv2.VideoCapture(video[num]))
# video = "../video/front1.mp4"
# video2 = "../video/front2.mp4"
# cap = cv2.VideoCapture(video)
# cap2 = cv2.VideoCapture(video2)
# frame=0
# while frame<len(path):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, path[frame][0])
#     cap2.set(cv2.CAP_PROP_POS_FRAMES, path[frame][1])
#     success, image = cap.read()
#     success2, image2 = cap2.read()
#     img = cv2.hconcat([image, image2]) 
#     if not (success or success2):
#         break
#     cv2.imshow("VideoFrame",img)
#     frame+=1
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
# cap.release()
