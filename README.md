## DTW-normalization
$ conda env create --f environment.yaml  

$ conda activate dtwNormal  

$ cd code  

$ python PyhtonMediapipe.py (videoPath)
ex) python PyhtonMediapipe.py ../video/ama_driver/

(video에 video들을 저장하고, videoPath, videoName, videoNum을 수정하고 실행)

현재 progirl_iron, progirl_driver  << 문제 큼
또한 1번을 기준으로 dtw를 돌려 1번이 평균과 가깝다고 뜰 확률 높음


|videoName|bestVid|CloseFrameCnt|
|---|---|---|
|ama_driver         |   1   |59	31	5	34	17	34|
|ama_iron           |   1   |49	44	33	19	24	11|
|amagirl_driver     |   1   |52	26	36	16	16	25	9|
|amagirl_iron       |   1   |97	23	17	11	21	11|
|pro1_driver        |   4   |17	39	17	46	19	19	10	13||
|pro1_iron          |   6   |30	36	12	13	13	41	20	15|
|pro2_driver        |   1   |102	37	6	8	12	11	4|
|pro2_iron          |   4   |45	0	22	63	12	6	24	8|
|pro3_driver        |   1   |40	25	23	21	14	35	12	10|
|pro3_iron          |   1   |55	22	13	21	2	16	24	27|
|progirl_driver     |   2   |4	66	24	10	12	7	12	16	9	20|
|progirl_iron       |   1   |122	4	5	4	36	1	5	3|


dtw를 video1기준으로 matching을 하기때문에 평균과 근사한 video로 1이 많이 검출됨. → 문제 해결 필요

progirl_driver과 같은 경우 (video1 frame, video2 frame)에 대해 (0:123,  0:41), (124:180, 115:180)과 같이
dtw가 매우 심각한 수준으로 진행되었으며 그런데도 1번 video가 아닌 2번 video를 평균으로 잡고 있음.
(아마 모든 video 중 2번 video가 outlier가 매우 심해서 2번으로 인해 평균값이 2번 video와 근접해져서
2번 video를 평균과 가장 근접하다고 잡을 가능성이 있음)

dtw link기준 frame이 20이상 차이나면 dtw검증이 제대로 되지 않았다고 판별할 경우
progirl_driver / progirl_iron / pro2_driver /  pro2_iron 이 제대로 판별되지 않음. 
(이는 mediapipe에서 data를 뽑아오는 과정에서 outlier이 많이 검출되어 현상 발생)
