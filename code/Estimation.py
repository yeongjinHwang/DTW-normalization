import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import norm
from numpy import dot
from PIL import Image
import time

results=1
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
def moving_average_x(idx, x,w):
  return np.convolve(x[idx][0], np.ones(w), 'valid') / w
def moving_average_y(idx, x,w):
  return np.convolve(x[idx][1], np.ones(w), 'valid') / w

class MesurePose ():
    def __init__(self):
        self.state = 0
        self.count = 0
        self.flag = 0
        self.timer = 0
        self.takeY1 = 0
        self.takeY2 = 0
        # self.start_time = time.time()
        self.start_time = 0
        self.IMAGE_FILES = []
        self.BG_COLOR = (192, 192, 192) # gray
        self.frame=[0,0,0,0,0,0,0]
        self.prevX=[0 for _ in range(33)]
        self.prevY=[0 for _ in range(33)]
        self.curX=[0 for _ in range(33)]
        self.curY=[0 for _ in range(33)]
        self.curV=[0 for _ in range(33)]
        self.ok15 = 0
        self.ok16 = 0
        self.smallX1 = 1.0
        self.smallY1 = 1.0
        self.smallX2 = 1.0
        self.smallY2 = 1.0
        self.largeX1 = 0.0
        self.largeX2 = 0.0
        self.images = 1
        self.totalTimer = 0
        # self.imgs = np.zeros((7,1080,640,3))
        self.imgs = np.zeros((7,960,540,3))
        self.totalTime = self.count
        self.totalFlag = 0


    def reliableCheck(self,idx):
        if abs(self.curX[idx[0]] - self.curX[idx[1]]) > 0.035 or abs(self.curY[idx[0]] - self.curY[idx[1]]) > 0.035:
            if self.curV[idx[0]] < self.curV[idx[1]]:
                self.curX[idx[0]] = self.curX[idx[1]]
                self.curY[idx[0]] = self.curY[idx[1]]
            else:
                self.curX[idx[1]] = self.curX[idx[0]]
                self.curY[idx[1]] = self.curY[idx[0]]

    ##==============================================================================


    def check_model_error(self,idx):
        if landmark_list == None:
            return 
        for i in range(0,33):
            for j in range (0,4):
                if self.data[i][0][j] == 0:
                    self.data[i][0][idx%4]=landmark_list.landmark[i].x
                    self.data[i][1][idx%4]=landmark_list.landmark[i].y
                    continue
            for j in range (0,4):
                if self.data[i][0][j] == 0:
                    continue
            avg_x = moving_average_x(i,self.data,4)
            avg_y = moving_average_y(i,self.data,4)
            if abs(avg_x -landmark_list.landmark[i].x) > 0.25 or abs(avg_y - landmark_list.landmark[i].y) > 0.25:
                landmark_list.landmark[i].x = avg_x
                landmark_list.landmark[i].y = avg_y
            else :
                self.data[i][0][idx%4]=landmark_list.landmark[i].x
                self.data[i][1][idx%4]=landmark_list.landmark[i].y


        ##==============================================================================
    def ImageSave(self,img):
        cv2.imwrite(self.fname+'/'+'0'+str(self.images)+'_'+str(self.state)+'.jpg',img)
        self.frame[self.state] = self.count
        # print(str(self.count) + "save Image" + str(self.state))

    def CheckForAdress(self,resized_image):
        if self.flag == 0:
            # self.start_time = time.time()
            self.start_time = self.count
            self.flag = 1
        # self.timer = time.time() - self.start_time
        self.timer = self.count - self.start_time
        # if self.timer > 0.5:
        if self.timer > 12:
            self.flag = 0
            self.totalTime = self.count
            self.ImageSave(resized_image)
            self.takeY1 = self.curY[16]
            self.takeY2 = self.curY[15]
            self.state = 1
    

    def CheckForTakeAway(self):
        if self.flag == 0:
            # self.start_time = time.time()
            self.start_time = self.count
            self.totalTime = self.count
            self.flag = 1
        # self.timer = time.time() - self.start_time
        self.timer = self.count - self.start_time
        # if self.timer > 2.7:
        if self.timer > 81:
            self.flag = 0
            self.state = 0
            print('Treset')
            

    def CheckForTop(self,resized_image):
        if self.flag == 0:
             # self.start_time = time.time()
            self.start_time = self.count
            self.flag = 1
        # self.timer = time.time() - self.start_time
        self.timer = self.count - self.start_time
        # if self.timer > 0.5:
        if self.timer > 10:
            self.flag = 0
            self.totalTime = self.count
            self.state = 4

    def CheckForFollowThrough(self,resized_image):
        if self.flag == 0:
             # self.start_time = time.time()
            self.start_time = self.count
            self.flag = 1
        # self.timer = time.time() - self.start_time
        self.timer = self.count - self.start_time
        # if self.timer > 0.5:
        if self.timer > 7:
            self.flag = 0
            self.totalTime = self.count
            self.state = 6

    def CheckForFinish(self,resized_image):
        if self.flag == 0:
            # self.start_time = time.time()
            self.start_time = self.count
            self.flag = 1
        # self.timer = time.time() - self.start_time
        self.timer = self.count - self.start_time
        # if self.timer > 0.5:
        if self.timer > 10:
            self.flag = 0
            self.state = 7
            print('Finish')

    def CheckForOneCycle(self):
        if self.totalFlag == 0:
            # self.start_time = time.time()
            self.start_time = self.count
            self.totalFlag = 1
        # self.timer = time.time() - self.start_time
        self.timer = self.count - self.start_time
        # if self.totalTimer > 2:
        if self.totalTimer > 60:
            self.totalFlag = 0
            self.state = 0
            print('Creset')

    def alert(self, message, image):
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.8
            fontColor = (0, 0, 255)
            thickness = 3
            lineType = 2
            cv2.putText(image,message, 
                        (15,60), 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType) 
    def main(self):            
        for numofVideo in range (1,8):
            self.fname = "ama_driver"
            cap = cv2.VideoCapture('cutVideo/'+self.fname+'/'+self.fname+str(numofVideo)+'.mp4')
            # cap = cv2.VideoCapture('cutVideo/'+self.fname+str(numofVideo)+'.mp4')
            # if numofVideo != 3 :
            #     continue
            with mp_pose.Pose(
                    min_detection_confidence=0.51,
                    model_complexity=1,
                    min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    success, image = cap.read()
                    self.count = self.count+1
                    # time.sleep(0.1)
                    
                    if not success:
                        # print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        if self.state == 6:
                            self.totalFlag = 0
                            self.state = 0
                            print(self.frame) 
                            # merge
                            for i in range (0,7):
                                filename = self.fname+'/'+ '0' + str(self.images)+'_'+str(i)+'.jpg'
                                # print(filename)
                                self.imgs[i] = cv2.imread(filename)
                            vis = np.concatenate((self.imgs[0], self.imgs[1],self.imgs[2],self.imgs[3],self.imgs[4],self.imgs[5],self.imgs[6]), axis=1)
                            cv2.imwrite(self.fname+'/output/'+ str(self.images)+'.png', vis)
                            # print('Done!')
                            self.images = self.images+1
                        self.state = 0
                        self.totalFlag = 0
                        self.flag = 0
                        self.count = 0
                        break
                        # continue
                    # if self.count <= 360 :
                    #     continue
                    # elif self.count >= 540:
                    #     continue
                    # print(image.shape)
                    croped_image1=image[0:image.shape[0],0:int(image.shape[1]/2)]
                    croped_image2=image[0:image.shape[0],int(image.shape[1]/2):image.shape[1]]
                    # croped_image=image[0:960,360:1080]
                    croped_image=croped_image1
                    scale_percent = 100 # percent of original size
                    # width = int(image.shape[1] * scale_percent / 100)
                    # height = int(image.shape[0] * scale_percent / 100)

                    ############################################################# cropeed_image
                    width = int(croped_image.shape[1] * scale_percent / 100)
                    height = int(croped_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized_image = cv2.resize(croped_image, dim, interpolation = cv2.INTER_AREA)
                    # resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    resized_image.flags.writeable = False
                    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    results = pose.process(resized_image)
                    # print(resized_image.shape)

                    if results.pose_landmarks == None:
                        continue
                    if results!=1 :
                        for i in range(0,33):
                            self.prevX[i]=results.pose_landmarks.landmark[i].x
                            self.prevY[i]=results.pose_landmarks.landmark[i].y
                    resized_image.flags.writeable = True
                    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        resized_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    # print(self.prevX[16])
                    for i in range(0,33):
                        self.prevX[i] = self.curX[i]
                        self.prevY[i] = self.curY[i]
                        self.curX[i]=results.pose_landmarks.landmark[i].x
                        self.curY[i]=results.pose_landmarks.landmark[i].y
                        self.curV[i]=results.pose_landmarks.landmark[i].visibility
                        # print(str(i) +' '+ str(results.pose_landmarks.landmark[i].x)+' '+ str(results.pose_landmarks.landmark[i].y)+' '+ str(results.pose_landmarks.landmark[i].visibility))
                    
                    # print(str(13) +' '+str(self.count) +' '+ str(results.pose_landmarks.landmark[13].visibility))
                    # print(str(15) +' '+str(self.count) +' '+ str(results.pose_landmarks.landmark[15].visibility))
                    # print(str(16) +' '+str(self.count) +' '+ str(results.pose_landmarks.landmark[16].visibility))
                    # # print(str(7) +' '+ str(self.curX[7]))
                    # time.sleep(0.1)

                    ###################################### address
                    if self.state == 0:
                        if abs(self.curX[16] - self.prevX[16]) < 0.005 or abs(self.curX[15] - self.prevX[15]) < 0.005:
                            if abs(self.curY[16] - self.prevY[16]) < 0.005 or abs(self.curY[15] - self.prevY[15]) < 0.005:
                                self.CheckForAdress(resized_image)
                                self.smallX2 = 1.0
                                self.smallX1 = 1.0
                            else:
                                self.state = 0
                                self.flag = 0
                        else:
                            self.state = 0
                            self.flag = 0
                    ####################################### take away
                    elif self.state == 1:
                        self.CheckForOneCycle()
                        self.CheckForTakeAway()
                        if self.takeY2 - 0.03 > self.curY[15]:
                            if self.smallX2 > self.curX[15]:
                                self.smallX2 = self.curX[15]
                                self.ImageSave(resized_image)
                                self.state = 2 
                                self.ok15 = 0
                        elif self.takeY1 - 0.03 > self.curY[16]:
                            if self.smallX1 > self.curX[16]:
                                self.smallX1 = self.curX[16]
                                self.ImageSave(resized_image)
                                self.state = 2 
                                self.ok15 = 0

                    ####################################### #half
                    if self.state == 2 :
                        self.reliableCheck([15,16])
                        if self.curV[16] < self.curV[15]:
                            if self.curX[15] < self.smallX2:
                                self.smallX2 = self.curX[15]
                                self.ImageSave(resized_image)
                                self.ok15 = 0
                                # print(self.count)
                            else :
                                self.ok15 = self.ok15 + 1
                        else : 
                            if self.curX[16] < self.smallX1:
                                self.smallX1 = self.curX[16]
                                self.ImageSave(resized_image)
                                self.ok15 = 0
                                # print(self.count)
                            else :
                                self.ok15 = self.ok15 + 1
                        if self.ok15 > 10:
                            self.ok15 = 0
                            self.smallY1 = 1.0
                            self.smallY2 = 1.0
                            self.state = 3 
                        
                    ###################################### top
                    if self.state == 3 :
                        # print("now ",self.count)
                        self.totalFlag = 0
                        # self.reliableCheck([15,16])
                        if self.curV[15] > self.curV[16]: 
                            if self.curY[15] < self.smallY2 :
                                self.smallY2 = self.curY[15]
                                self.largeX1 = 0.0
                                self.largeX2 = 0.0
                                self.flag = 0
                                self.ImageSave(resized_image)
                        else :
                            if self.curY[16] < self.smallY1 :
                                self.smallY1 = self.curY[16]
                                self.largeX1 = 0.0
                                self.largeX2 = 0.0
                                self.flag = 0
                                self.ImageSave(resized_image)
                        self.CheckForTop(resized_image)

                    ##################################### impact
                    if self.state == 4 :
                        # self.reliableCheck([15,16])
                        # if self.curY[19] > self.curY[23] and self.curY[20] > self.curY[24]:
                        #     self.ImageSave(resized_image)
                        # if self.count >= 89:
                        if self.count >= 118:
                            self.ImageSave(resized_image)
                            self.state = 5
                    ##################################### follow through
                    if self.state == 5 :
                        # time.sleep(0.1)
                        # if self.takeY1 - 0.04 > self.curY[16]:
                        #     if self.largeX1 < self.curX[16]:
                        #             self.largeX1 = self.curX[16]
                        #             self.ImageSave(resized_image)
                        #             self.smallX1 = 1.0
                        #             self.state = 6 
                        # elif self.takeY2 - 0.04 > self.curY[15]:
                        #     if self.largeX2 < self.curX[15]:
                        #             self.largeX2 = self.curX[15]
                        #             self.ImageSave(resized_image)
                        #             self.smallX2 = 1.0
                        #             self.state = 6 
                        if self.largeX1 < self.curX[16]:
                                self.largeX1 = self.curX[16]
                                self.smallX1 = 1.0
                                self.ImageSave(resized_image)
                                self.flag = 0
                        elif self.largeX2 < self.curX[15]:
                                self.largeX2 = self.curX[15]
                                self.smallX2 = 1.0
                                self.ImageSave(resized_image)
                                self.flag = 0
                        else :
                            self.CheckForFollowThrough(resized_image)
                    #################################### finish
                    if self.state == 6 :
                        # if self.smallX1 > self.curX[16]:
                        #     self.smallX1 = self.curX[16]
                        #     self.flag = 0
                        # elif self.smallX2 > self.curX[15]:
                        #     self.smallX2 = self.curX[15]
                        #     self.flag = 0
                        # else :
                        #     self.CheckForFinish(resized_image)
                        self.reliableCheck([15,16])
                        # if abs(self.curX[16] - self.prevX[16]) < 0.002 or abs(self.curX[15] - self.prevX[15]) < 0.002:
                        #     if abs(self.curY[16] - self.prevY[16]) < 0.002 or abs(self.curY[15] - self.prevY[15]) < 0.002:
                        if self.curX[16] < self.largeX1 :
                            self.largeX1 = self.curX[16]
                            self.ImageSave(resized_image)
                            self.CheckForFinish(resized_image)  
                        elif self.curX[15] < self.largeX2 :
                            self.largeX2 = self.curX[15]
                            self.ImageSave(resized_image)
                            self.CheckForFinish(resized_image)  
                            #     self.ImageSave(resized_image)
                            #     self.CheckForFinish(resized_image)                                
                            # else:
                            #     self.state = 6
                            #     self.flag = 0
                        else:
                            self.state = 6
                            self.flag = 0
                    #################################### save
                    if self.state == 7:
                        self.totalFlag = 0
                        self.state = 0
                        print(self.frame) 
                        # merge
                        for i in range (0,7):
                            filename = self.fname+'/'+ '0' + str(self.images)+'_'+str(i)+'.jpg'
                            # print(filename)
                            self.imgs[i] = cv2.imread(filename)
                        vis = np.concatenate((self.imgs[0], self.imgs[1],self.imgs[2],self.imgs[3],self.imgs[4],self.imgs[5],self.imgs[6]), axis=1)
                        cv2.imwrite(self.fname+'/output/'+str(self.images)+'.png', vis)
                        # print('Done!')
                        self.count = 0
                        self.images = self.images+1
                    # Flip the image horizontally for a selfie-view display.
                    self.alert(str(self.count),resized_image)
                    cv2.imshow('MediaPipe Pose', resized_image)
                    # if self.count >= 118 and self.count <= 128:
                    #     time.sleep(0.1)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
            cap.release()

measure = MesurePose()
measure.main()