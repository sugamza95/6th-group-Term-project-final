"""trt_yolo_with_centernet.py
This script demonstrates how to do real time object detection using TensorRT optimized YOLO engine..
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt

import sys
import torchvision.transforms as transforms
import PIL.Image
import csv

from operator import itemgetter
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from matplotlib import pyplot as plt
WINDOW_NAME = 'TrtYOLODemo'



def get_keypoint(humans, hnum, peaks, people_num):
    """
        check the number of people using trt_pose keypoints
        only the objects over 6 keypoints out of 12 pass the test
    """
    kpoint = []
    human = humans[0][hnum]
    human_part_count = 0
    human_count_flag = 0
    C = human.shape[0]
    people_num_tmp = people_num
    for j in range(C):
        k = int(human[j]) 
        if k >= 0:
            human_part_count = human_part_count + 1
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            if human_part_count >= 6:
                human_count_flag = 1
            kpoint.append(peak)
        else:
            peak = (j, None, None)
            kpoint.append(peak)
        
    if human_count_flag == 1:
        people_num_tmp =  people_num_tmp + 1
        human_part_count = 0    
        return kpoint, human_count_flag, people_num_tmp
    else:
        return [], human_count_flag, people_num_tmp

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print('------ model = resnet--------')
MODEL_WEIGHTS = '/home/ee201511400/for_term/ee201511281/tensorRT-yolo/tensorrt_demos/resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = '/home/ee201511400/for_term/ee201511281/tensorRT-yolo/tensorrt_demos/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
WIDTH = 224
HEIGHT = 224
'''
print('------ model = densenet--------')
MODEL_WEIGHTS = '/home/ee201511400/for_term/ee201511281/tensorRT-yolo/tensorrt_demos/densenet121_baseline_att_256x256_B_epoch_160.pth'
OPTIMIZED_MODEL = '/home/ee201511400/for_term/ee201511281/tensorRT-yolo/tensorrt_demos/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
WIDTH = 256
HEIGHT = 256
'''

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()
print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

font = cv2.FONT_HERSHEY_PLAIN        
line = cv2.LINE_AA

#the rectangle contour of hand sanitizer
BOX_1_WIDTH_START = 250
BOX_1_WIDTH_END = 400
BOX_1_HEIGHT_START = 200
BOX_1_HEIGHT_END = 450

BOX_2_WIDTH_START = 650
BOX_2_WIDTH_END = 800
BOX_2_HEIGHT_START = 200
BOX_2_HEIGHT_END = 450
    
def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
def sort_return_time(box_class,id_position,time):
   
    
    if 0 not in id_position:
        id_position = sorted(id_position, key=itemgetter(0))
    if(len(id_position) == 1):#for recognition errors
        for i in range(len(box_class)-len(id_position)):
            if(id_position[0][3] == 'left'):
                id_position = [id_position[0],0]
            elif(id_position[0][3] == 'right'):
                id_position = [0,id_position[0]]
    for i in range(len(box_class)):# limited for the person who wears mask
        if(len(id_position) ==0):
            break
        if(id_position[i]):
            if(round(box_class[i][1]) == 1 and ((id_position[i][1]<BOX_1_WIDTH_END and id_position[i][1]>BOX_1_WIDTH_START and id_position[i][2] >BOX_1_HEIGHT_START and id_position[i][2] <BOX_1_HEIGHT_END) or (id_position[i][1]> BOX_2_WIDTH_START and id_position[i][1] < BOX_2_WIDTH_END and id_position[i][2] >BOX_2_HEIGHT_START and id_position[i][2] < BOX_2_HEIGHT_END))):
                time[i]=1
            else:
                time[i]=0
        else:
            time[i]=2 # 2 for not recognized(skip dicision)
    return id_position, time

    
def execute(img, src, people_num,id_position,box_class,cam):
    color = (0, 255, 0)
    id_position=[]
    time = [2,2]
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf) #, cmap_threshold=0.15$
    people_num_tmp = people_num
    keypoints_agg=[]
    for i in range(counts[0]):
        keypoints, human_flag, people_num_tmp = get_keypoint(objects, i, peaks, people_num_tmp)
        if(human_flag == 1):
            keypoints_agg.append(keypoints)
    #if the # of people is 1 
    if(len(keypoints_agg) == 1):
        if(len(box_class) != 0):
            if(abs(box_class[0][0]-BOX_1_WIDTH_START) < abs(box_class[0][0]-BOX_2_WIDTH_START)):
                if(keypoints_agg[0][10][1] is not None):
                    x1 = round(keypoints_agg[0][10][2]*cam.img_width)
                    y1 = round(keypoints_agg[0][10][1]*cam.img_height)
                    id_position = [[box_class[0][0],x1,y1],0]
                   # cv2.circle(src, (x1,y1), 3, color, 2)
                elif(keypoints_agg[0][9][1] is not None):
                    x1 = round(keypoints_agg[0][9][2]*cam.img_width)
                    y1 = round(keypoints_agg[0][9][1]*cam.img_height)
                    id_position = [[box_class[0][0],x1,y1],0]
                    #cv2.circle(src, (x1,y1), 3, color, 2)
                box_class=[box_class[0],[0,0]]

            else:
                if(keypoints_agg[0][9][1] is not None):
                    x1 = round(keypoints_agg[0][9][2]*cam.img_width)
                    y1 = round(keypoints_agg[0][9][1]*cam.img_height)
                    id_position = [0,[box_class[0][0],x1,y1]]
                    #cv2.circle(src, (x1,y1), 3, color, 2)
                elif(keypoints_agg[0][10][1] is not None):
                    x1 = round(keypoints_agg[0][10][2]*cam.img_width)
                    y1 = round(keypoints_agg[0][10][1]*cam.img_height)
                    id_position = [0,[box_class[0][0],x1,y1]]
                   # cv2.circle(src, (x1,y1), 3, color, 2)
                box_class=[[0,0], box_class[0]]

    #if the # of people is 2 
    elif(len(keypoints_agg) == 2):
     for j in range(len(keypoints_agg)):
        if(keypoints_agg[j][11][1] is not None):# right hip
            x = round(keypoints_agg[j][11][2]*cam.img_width)
            y = round(keypoints_agg[j][11][1]*cam.img_height)
            #cv2.circle(src, (x, y), 3, color, 2) 
            #rihgt
            if(len(box_class) == 2):
             if(abs(box_class[0][0]-x) < abs(box_class[1][0]-x)):
                if(keypoints_agg[j][10][1] is not None):
                 x1 = round(keypoints_agg[j][10][2]*cam.img_width)
                 y1 = round(keypoints_agg[j][10][1]*cam.img_height)
                 id_position.append([x,x1,y1,'left']) #[hip position, wrist position]
                 #cv2.circle(src, (x1,y1), 3, color, 2)
            #left
             elif(abs(box_class[0][0]-x) > abs(box_class[1][0]-x)):
                if(keypoints_agg[j][9][1] is not None):
                 x1 = round(keypoints_agg[j][9][2]*cam.img_width)
                 y1 = round(keypoints_agg[j][9][1]*cam.img_height)
                 id_position.append([x,x1,y1,'right']) #[hip position, wrist position]
                 #cv2.circle(src, (x1,y1), 3, color, 2)
    
    #sort x coordinate ascending
    id_position,time = sort_return_time(box_class,id_position,time)
    
    return people_num_tmp,id_position,time

X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args

def get_sorted_box(boxes,clss):
        _box=[]
        _clss=[]
        for box in boxes:
            _box.append(box[0])
        for clss in clss:
            _clss.append(clss)
        std_box_class=[] # sort box ascending(the position of array is id)
        for box,clss in zip(_box,_clss):
            std_box_class.append([box,clss])
        std_box_class.sort(key=lambda x:x[0])
        return std_box_class, _clss

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.
    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    time_tmp = 0
    count_t = [0,0]
    flag=[0,0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('/home/ee201511281/sample_mask_test.mp4', fourcc, 30.0 , (cam.img_width, cam.img_height))
    
    full_scrn = False
    fps = 0.0
    tic = time.time()
    people_num =0 
    id_position=[]
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        resize = cv2.resize(img,(640,480))
        img_temp = img
        boxes, confs, clss, no_mask_count, mask_count = trt_yolo.detect(img, conf_th)
        std_box_class,_clss = get_sorted_box(boxes,clss)
        
        img_e = cv2.resize(img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        people_num,id_position,time_for_disinfection  = execute(img_e, img, people_num,id_position,std_box_class,cam)

        # handling all the return values
        if no_mask_count == 0:
          if(len(time_for_disinfection) == 2):
            for index, el in enumerate(time_for_disinfection):
                if(el == 1):
                    count_t[index]+=1
                elif(el == 0):
                    count_t[index] = 0
                else:
                    count_t[index] = count_t[index]
          for index, el in enumerate(count_t):
            if(el >= 20):
                flag[index] = 1
          print("time_count", count_t);
          for index, el in enumerate(flag):
            if(len(id_position) is not 0 and id_position[index] is not 0 and people_num==1):
                if(el == 1):
                    if(index == 1):
                        cv2.putText(img, "You are allowed to enter", (400,50), font, 1.0,(32,32,32), 1, line)
                    else:
                         cv2.putText(img, "You are allowed to enter", (200,50), font, 1.0,(32,32,32), 1, line)

                else:
                    if(index == 1 ):
                        cv2.putText(img, "You are not allowed to enter", (400,50), font, 1.0,(32,32,32), 1, line)
                    else:
                         cv2.putText(img, "You are not allowed to enter", (200,50), font, 1.0,(32,32,32), 1, line)
                


            elif(len(std_box_class)==2):
              if(el == 1):
                cv2.putText(img, "You are allowed to enter", (std_box_class[index][0],50), font, 1.0,(32,32,32), 1, line)
                
              else:
                 cv2.putText(img, "You are not allowed to enter", (std_box_class[index][0],50), font, 1.0,(32,32,32), 1, line)
        else:
            cv2.putText(img, "Put on your mask!", (50, 300), font, 2.0, (0,0,255),2, line)
        if people_num == 0:
            count_t = [0,0]
            flag=[0,0]
            cv2.putText(img, "Now you can come in", (50,300), font, 2.0, (255,0,0), 2, line)
        img = vis.draw_bboxes(img, boxes, confs,_clss)
        img = show_fps(img, fps)
        cv2.putText(img, 'mask = ' + str(mask_count), (11, 300), font, 3.0, (32, 32, 32), 4, line)
        cv2.putText(img, 'mask = ' + str(mask_count), (10, 300), font, 3.0, (240, 240, 240), 1, line)
        cv2.putText(img, 'peaple = ' + str(people_num), (11, 400), font, 3.0, (32, 32, 32), 4, line)
        cv2.putText(img, 'peaple = ' + str(people_num), (10, 400), font, 3.0, (240, 240, 240), 1, line)
        people_num = 0
        cv2.imshow(WINDOW_NAME, img)        
        out_video.write(img)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)

        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        print('FPS = %d\n'%(fps))
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
    f.close() 
    out_video.release()
    #end

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    
       
    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)

    open_window(
         WINDOW_NAME, 'Camera TensorRT YOLO Demo',
         640, 480)

    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
