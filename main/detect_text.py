import argparse
import os
import sys
import time
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append(os.getcwd())

from utils.text_connector.detectors import TextDetector
from utils.rpn_msr.proposal_layer import proposal_layer
from nets import model_train as model

parser = argparse.ArgumentParser(
    description='module take receipt path and returns list of predicted bounding boxes for every text region')
parser.add_argument('--image', help='path of the receipt image')
parser.add_argument('--debug', help='bolean variable to show predictions')
args = parser.parse_args()
tf.app.flags.DEFINE_string('debug', args.debug, '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS





def show_img(img, title="test"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def sort_boxes(pts):
    return np.array(sorted(pts, key=lambda k: [k[1], k[0]]))

def remove_shadow(image):
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 15)
        diff_img = 255 - cv2.absdiff(img, bg_img)
        norm_img = diff_img.copy()  # Needed for 3.x compatibility
        cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
        cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        thr_img = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2BGR)
        return thr_img



def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    if im_scale >= 1:
        # img = cv2.resize(img, dim, interpolation= cv2.INTER_CUBIC)
        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        # img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)
        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

def show_img(img, title=""):
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 600,600)
    cv2.imshow(title, img)
    cv2.waitKey(0)

def get_mser(img):
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
    gray_img = img.copy()
    regions, _  = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(gray_img, hulls, 1, (0, 0, 255), 2)
    show_img(gray_img)

def remove_shadow(img):

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result

class DetectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
        elif len(self.src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15
        h_size = int(h_img.shape[1] / scale)

        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        h_erode_img = cv2.erode(h_img, h_structure, 1)

        h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)
        # cv2.imshow("h_erode",h_dilate_img)
        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

        mask_img = h_dilate_img + v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
        # cv2.imshow("joints", joints_img)
        # cv2.imshow("mask", mask_img)

        return mask_img, joints_img

def table_removal(procced_img):
    mask, joint = DetectTable(procced_img).run()
    kernel = np.ones((5, 5), np.uint8)
    if len(procced_img.shape) == 3 and procced_img.shape[2] == 3:
        mask_dilation = cv2.dilate(mask, kernel, iterations=1)
        mask_dilation = np.stack((mask_dilation,) * 3, axis=-1)
        dst = cv2.add(procced_img, mask_dilation)
        return dst
    elif len(procced_img.shape) == 2 or procced_img.shape[2] == 1:
        mask_dilation = cv2.dilate(mask, kernel, iterations=1)
        dst = cv2.add(procced_img, mask_dilation)
        return dst

def remove_bg(img):
    # Original Code
    CANNY_THRESH_2 = 200

    # Change to
    CANNY_THRESH_2 = 100

    ####### Change below worth to try but not necessary

    # Original Code
    mask = np.zeros(img.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # Change to
    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))
    show_img(remove_bg)
    
def adaptive_threshold(img):
    
    # Convert Image To Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
    
    # Remove Shadow from image
    # img = remove_shadow(img)
    # Remove Tables
    # img = table_removal(img)
    # Heighlight text regions
    img = cv2.erode(img,None,iterations = 2)
    show_img(img, "Erode")
    # Binarize image 
    # ret2,img = cv2.threshold(img, 10,255,cv2.THRESH_OTSU)
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 0)

    # kernel = np.ones((5,5),np.float32)/25
    # img = cv2.filter2D(img,-1,kernel)
    contours, hierarchy = cv2.findContours(img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for cnt in contours:
        # get convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img, [hull], -1, (0, 0, 255), 1)  

    show_img(img,"After thre")
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # img = cv2.adaptiveThreshold(img,255,1,1,31,2)   
    # show_img(img,"After threshold")
    # kernel = np.ones((5,5),np.float32)/25
    
    # img = cv2.filter2D(img,-1,kernel)
    # show_img(img,"After kernal")
    # show_img(img,"after")
    # Find the contours
    # contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # # For each contour, find the bounding rectangle and draw it
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #     # cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
    # show_img(img,"after")
    # kernel = np.ones((2,2),np.float32)/4
    # show_img(img,"before")
    # img = cv2.filter2D(img,-1,kernel)
    # show_img(img,"after")
    return img

def deskew_image(img, boxes):
    angle_acc = 0
    for i, box in enumerate(boxes): 
        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect) 
        box = np.int0(box)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        # else:
        #     angle = -angle
        angle_acc += angle
    angle_acc /= len(boxes)
   
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_acc, 1.0)
    try:
        img = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  
    except:
        pass

    return img

def crop_image(img, boxes, write_image=True, verbose=False):

    # Sort boxes ascending () Topmost point then Leftmost point )
    boxes = np.array( sorted(boxes , key=lambda k: [k[1], k[0]]) )

    # Extract interset points to crop receipt
    leftmost = max(0, min([ min(boxes[:,0]), min(boxes[:,6])]) ) # max(0,number) to avoid -1 returning
    rightmost = max([ max(boxes[:,2]), max(boxes[:,4])])
    topmost = max(0, min([ min(boxes[:,1]), min(boxes[:,3])])  ) # max(0,number) to avoid -1 returning
    bottommost = max([ max(boxes[:,5]), max(boxes[:,7])])

    # Reshape the interset points to the following shape [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    pts = np.array([leftmost, topmost, rightmost,topmost,rightmost, bottommost, leftmost, bottommost])\
                .astype(np.int32).reshape((-1, 2))
    

    # Create the receipt bounding rectangle from interset points
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = img[y:y+h, x:x+w]

    return cropped, pts
    

    if write_image:
        cv2.imwrite(os.path.join(FLAGS.output_path, "cropped_" + img_name.replace('jpeg','tiff').replace('jpg','tiff')), cropped[:, :, ::-1])



def detect_text(img, sess,bbox_pred, cls_pred, cls_prob,input_image,input_im_info,mode='O'):
    
    start = time.time()
    if len(img.shape) == 3:
        h, w, c = img.shape 
    else:
        c = 1
        h, w = img.shape 

    im_info = np.array([h, w, c]).reshape([1, 3])   
    
    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                            feed_dict={input_image: [img],
                                                        input_im_info: im_info})

    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]
    textdetector = TextDetector(DETECT_MODE=mode)
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])

    boxes = np.array(boxes, dtype=np.int)

    
    return img, boxes

def visualize_detection(orig_image, boxes,title):   
    img = orig_image.copy()
    color = list(np.random.choice(range(256), size=3))
    clrList = [(0,0,255), (0,255,0), (255,0,0),(255,255,0),(255,0,255),(0,255,255)]
    for i, box in enumerate(boxes):

        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color=clrList[int(i%len(clrList))], thickness=3)

    show_img(img, title)

def merge_boxes(boxes):

    threshold = 10
    merge_count = 0
    black_list = []
    new_boxes = []
    for i, box in enumerate(boxes):                   
        # Skip the merged boxes
        if i in black_list:
            continue
        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, color=(0, 0, 255), thickness=2)
        # show_img(img)
       
        # Loop on all boxes after current box
        for idx in range(i+1, len(boxes)):
            # Skip the merged boxes
            if idx in black_list:
                continue
            # Set temp_box as the next box
            tmp_box = boxes[idx]
            # Check if Height difference - of one of two corners - less than threshold (i.e the same line)
            if abs(tmp_box[1] - box[1]) < threshold or abs(tmp_box[3] - box[3]) < threshold \
                or abs(tmp_box[5] - box[5]) < threshold or abs(tmp_box[7] - box[7]) < threshold:
                black_list.append(idx)
                # count how many boxes are merged
                merge_count = merge_count + 1
                # stretch the original width box to cover the two boxes (Consider stretching from LTR or RTL)
                box[0] = min(tmp_box[0], box[0] )
                box[6] = min(tmp_box[6], box[6])
                box[2] = max(tmp_box[2], box[2])
                box[4] = max(tmp_box[4], box[4])
                # selecet the largest height and set the original box to the larger one (to avoid clipping)
                max_height_left_corner = np.min( [box[1], box[3], tmp_box[1],tmp_box[3]])                          
                box[1] = box[3] = max_height_left_corner
                # selecet the largest lower height and set the original box to the larger one (to avoid clipping)
                max_height_right_corner =np.max(  [box[5], box[7], tmp_box[5],tmp_box[7]] )                           
                box[5] = box[7] = max_height_right_corner
        
        # box[0] = box[6] = leftmost
        # box[2] = box[4] = rightmost
        new_boxes.append(box)
        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
    new_boxes = np.array( sorted(new_boxes , key=lambda k: [k[1], k[0]]) )
    return new_boxes

def sub_line_equation( x1,y1 ,x2,y2 , x=None, y=None):
    m = (y1 - y2) / (x1 - x2)
    if y is None:
        y_calc = m * (x - x1) + y1
        return y_calc
    elif x is None:
        x_calc = ((y - y1) / m) + x1
        return x_calc

    return (x_calc + x,y_calc + y)

def get_relative_distance(orig_pts, boxes):
    line1 = np.reshape([orig_pts[0] , orig_pts[1]], -1)
    line2 = np.reshape([orig_pts[0] , orig_pts[3]], -1)
    for idx, box in enumerate(boxes):
        box = box[:8].astype(np.int32).reshape((-1, 2))
        for i in range(0,8,2):
            boxes[idx][i] = boxes[idx][i] + sub_line_equation(line2[0],line2[1],line2[2],line2[3], y=boxes[idx][i+1])
            boxes[idx][i+1] = boxes[idx][i+1] + sub_line_equation(line1[0],line1[1],line1[2],line1[3], x=boxes[idx][i])
    return boxes   

def crop_boxes(img,boxes):
    lines = []
    for i, box in enumerate(boxes):    
        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
        pts[pts<0] = 0

        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], color=(255,255,255))
        # show_img(mask, "detect_text-mask")
        res = cv2.bitwise_and(img, img, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        inv_mask = cv2.merge((inv_mask,inv_mask,inv_mask))
        res = res + inv_mask
  
        line_rect = cv2.boundingRect(pts)
        x, y, w, h = line_rect
        croped_line = res[y:y+h, x:x+w].copy()       
        croped_line = deskew_image(croped_line, [box])
        # show_img(croped_line, "detect_text-croped_line")
        # croped_line = cv2.cvtColor(croped_line, cv2.COLOR_BGR2GRAY)
        # _,croped_line = cv2.threshold(croped_line, 0,255,cv2.THRESH_OTSU)   
        # croped_line = cv2.cvtColor(croped_line,cv2.COLOR_GRAY2RGB)
        lines.append(croped_line)
    return lines

def stretch_boxes(input_img_shape, resized_image_shape, boxes):
    input_h, input_w = input_img_shape[0:2]
    resized_h, resized_w = resized_image_shape[0:2]
    ratio_w = (input_w / resized_w)
    ratio_h = (input_h / resized_h)   
    for box in boxes:
        box[0] *= ratio_w
        box[2] *= ratio_w
        box[4] *= ratio_w
        box[6] *= ratio_w

        box[1] *= ratio_h
        box[3] *= ratio_h
        box[5] *= ratio_h
        box[7] *= ratio_h
    return boxes




def main(receipts):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')
            

        try:
            global_step = tf.get_variable(
                'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        except: pass
        bbox_pred, cls_pred, cls_prob = model.model(input_image) 

        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        final_results = []
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(
                ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            start = time.time()

            for idx, receipt in enumerate(receipts):
                # if idx<4: continue
                print("Receipt Number %d From %d" % (idx+1, len(receipts)))
                tic = time.time()
                input_img = cv2.imread(receipt) if isinstance(receipt, str) else receipt
                text_regions = []
                #####################> First Stage <##########################
                resized_img, (rh, rw) = resize_image(input_img)   
                resized_img, boxes = detect_text(resized_img, sess, bbox_pred, cls_pred, cls_prob,input_image,input_im_info)
                boxes = stretch_boxes(input_img.shape,resized_img.shape, boxes)
                visualize_detection(input_img, boxes, "First Stage (Line Detection Only)")
                # continue
                #####################> 2nd Stage <##########################
                input_img, orig_pts = crop_image(input_img, boxes, False)
                resized_img, (rh, rw) = resize_image(input_img)  
                img, boxes = detect_text(resized_img, sess, bbox_pred, cls_pred, cls_prob,input_image,input_im_info,mode='H')
                # boxes = stretch_boxes(input_img.shape,resized_img.shape, boxes)
                # visualize_detection(img, boxes, "Second Stage (Receipt Cropping -> Line Detection)")
                boxes = merge_boxes(boxes)
                # visualize_detection(img, boxes, "merged")
                #####################> 3rd Stage <##########################
                # input_img, orig_pts = crop_image(input_img, boxes, False)
                # input_img = deskew_image(input_img, boxes)
                # resized_img, (rh, rw) = resize_image(input_img)  
                # img, boxes = detect_text(resized_img, sess, bbox_pred, cls_pred, cls_prob,input_image,input_im_info)
                # boxes = stretch_boxes(input_img.shape,resized_img.shape, boxes)
                # visualize_detection(input_img, boxes, "3rd Stage (Receipt Cropping -> Deskew -> Line Detection)")
                ############################################################################
                boxes = sort_boxes(boxes)
                lines = crop_boxes(img, boxes)
                final_results.append(np.array(lines))
                toc = time.time()
                print("cost time: {:.2f}s".format(toc-tic))
                print("======================")

    # tf.reset_default_graph()
    return final_results
            # return text_regions
    tf.app.run()

def crop_deskew(img, points):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], color=(255,255,255))
    show_img(mask, "detect_text-mask")
    res = cv2.bitwise_and(img, img, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    inv_mask = cv2.merge((inv_mask,inv_mask,inv_mask))
    res = res + inv_mask
    
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    # try:
    #     gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # except:
    #     return None

    # gray = cv2.bitwise_not(gray)
    # show_img(gray, 'gray') # white foreground and black background

    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    # Taking a matrix of size 3 as the kernel 
    # kernel = np.ones((3,3), np.uint8) 
    # img_erosion = cv2.erode(thresh, kernel, iterations=1)
    # show_img(img_erosion, 'img_erosion') # connected white foreground and black background

    # coords = np.column_stack(np.where(thresh > 0))
    # angle = cv2.minAreaRect(coords)[-1]

    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle

    # # print(angle)

    # # rotate the image to deskew it
    # (h, w) = cropped.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(cropped, M, (w, h),
    #                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    
    # show_img(cropped, 'crop')
    # show_img(rotated, 'rotated')
    show_img(cropped, "detect_text-crop_deskew")
    return cropped
if __name__ == '__main__':
    main(receipts=['data/demo/IMG_20190129_135319.jpg'])
