
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as pth
from matplotlib.collections import PatchCollection


class Simple_tracker():
    def __init__(self, resolution, iou_thre=0.1, occlude_delay=10):
        super().__init__()
        self.res = resolution # (width x height) in px
        self.thre = iou_thre
        self.delay = occlude_delay # after 'delay' time the obj is regarded died
        self.tracking_obj = pd.DataFrame({'ID':[0], 'Frame':0, 'xmin':0, 'ymin':0, 'xmax':0, 'ymax':0, 'Conf':0, 'Label':'test', 'Occlude':0})

    def obj_tracking(self, boxes, frame):
        assert (frame<1e4),('Time out {}.'.format(frame))

        for i in range(len(boxes[0])):
            boxes[0][i][0] = boxes[0][i][0]*self.res[0]
            boxes[0][i][1] = boxes[0][i][1]*self.res[1]
            boxes[0][i][2] = boxes[0][i][2]*self.res[0]
            boxes[0][i][3] = boxes[0][i][3]*self.res[1]
        boxes = self.current_frame_analyze(boxes) # get all objs in the current frame as list([[[sample]]])
        obj_last = self.tracking_obj.loc[self.tracking_obj['Frame']==(frame-1),:] # get all objs in the last frame as df
        obj_last = obj_last.reset_index(drop=True)
        if obj_last.empty | (frame-1)==0:
            id = max(self.tracking_obj.loc[:,'ID'])
            for box in boxes[0]:
                id += 1
                df = self.box2df(box, id, frame)
                self.tracking_obj = self.tracking_obj.append(df, ignore_index=True)
        else:
            id = max(self.tracking_obj.loc[:,'ID'])
            no_match = list(range(obj_last.shape[0])) # the list of obj from last frame, which has no match in this frame
            for i in range(len(boxes[0])): # iterate over all objs from the this frame
                boxes_last = []
                for j in range(obj_last.shape[0]):
                    boxes_last.append(self.df2box(obj_last.loc[j,:]))
                boxes_last = [boxes_last] # df2boxes from last frame

                idx = self.data_association(boxes[0][i], boxes_last) # return last idx, if no, return -1
                if (id+1+idx) == id: # new objects
                    id += 1
                    df = self.box2df(boxes[0][i], id, frame)
                else: # existing objects
                    df = self.box2df(boxes[0][i], obj_last.loc[idx,'ID'], frame)
                    no_match.remove(idx)
                self.tracking_obj = self.tracking_obj.append(df)
            for i in no_match:
                if obj_last.loc[i,'Occlude']<self.delay:
                    df = obj_last.loc[i,:].copy()
                    df.loc['Occlude'] = frame-df.loc['Frame']+df.loc['Occlude']
                    df.loc['Frame'] = frame
                    self.tracking_obj = self.tracking_obj.append(df)
        self.tracking_obj = self.tracking_obj.reset_index(drop=True)

    def plot_tracking(self, frame=-1):
        assert (frame<=max(self.tracking_obj.loc[:,'ID'])),('Frame number exceeds the maximum.')
        fig, ax = plt.subplots(1)
        if frame>0:
            df = self.tracking_obj.loc[self.tracking_obj['Frame']==frame,:]
            boxes = []
            for i in range(df.shape[0]):
                boxes.append(self.df2box(df.iloc[i,:]))
                x0 = (df.iloc[i,:]['xmin'] + df.iloc[i,:]['xmax']) / 2
                y0 = (df.iloc[i,:]['ymin'] + df.iloc[i,:]['ymax']) / 2
                plt.text(x0, y0, str(df.iloc[i,:]['ID']))
            recs = [pth.Rectangle((box[0],box[1]), width=(box[2]-box[0]), height=(box[3]-box[1])) for box in boxes]
            ax.add_collection(PatchCollection(recs))
            ax.set_xlim([0,self.res[0]])
            ax.set_ylim([0,self.res[1]])
            ax.axis('equal')
            plt.show()
        else: # show all history
            frame_list = list(set(self.tracking_obj['Frame'].values.tolist()))
            c = [str(round((255-x)/255,1)) for x in np.linspace(1,255,len(frame_list))]
            alp = np.linspace(0,1,len(frame_list))
            for f in frame_list: # iterate over all frames
                df = self.tracking_obj.loc[self.tracking_obj['Frame']==f,:]
                boxes = []
                for j in range(df.shape[0]): # iterate over all objs in this frame
                    boxes.append(self.df2box(df.iloc[j,:]))
                    x0 = (df.iloc[j,:]['xmin'] + df.iloc[j,:]['xmax']) / 2
                    y0 = (df.iloc[j,:]['ymin'] + df.iloc[j,:]['ymax']) / 2
                    plt.text(x0, y0, str(df.iloc[j,:]['ID']), color='r')

                    # plt.arrow(x0, y0, dx, dy)
                recs = [pth.Rectangle((box[0],box[1]), width=(box[2]-box[0]), height=(box[3]-box[1]), color=c[f]) for box in boxes]
                ax.add_collection(PatchCollection(recs, match_original=True, edgecolor='b', alpha=alp[f]))
            ax.set_xlim([0,self.res[0]])
            ax.set_ylim([0,self.res[1]])
            ax.axis('equal')
            plt.show()

    def plot_tracking_onimage(self, frame, img):
        fig, ax = plt.subplots(1)
        ax.imshow(img)  
        df = self.tracking_obj.loc[self.tracking_obj['Frame']==frame,:]
        boxes = []
        for i in range(df.shape[0]):
            boxes.append(self.df2box(df.iloc[i,:]))
            x0 = (df.iloc[i,:]['xmin'] + df.iloc[i,:]['xmax']) / 2
            y0 = (df.iloc[i,:]['ymin'] + df.iloc[i,:]['ymax']) / 2
            plt.text(x0, y0, str(df.iloc[i,:]['ID']))
            plt.text(df.iloc[i,:]['xmin'], df.iloc[i,:]['ymin'], str(df.iloc[i,:]['Label']))
        recs = [pth.Rectangle((box[0],box[1]), width=(box[2]-box[0]), height=(box[3]-box[1]), facecolor='none', edgecolor='r') for box in boxes]
        ax.add_collection(PatchCollection(recs, match_original=True,))
        ax.set_xlim([0,self.res[0]])
        ax.set_ylim([0,self.res[1]])
        ax.axis('equal')
        plt.show()
          

    def df2box(self, dataSeries):
        xmin = dataSeries.loc['xmin']
        ymin = dataSeries.loc['ymin']
        xmax = dataSeries.loc['xmax']
        ymax = dataSeries.loc['ymax']
        conf = dataSeries.loc['Conf']
        label = dataSeries.loc['Label']
        return [xmin, ymin, xmax, ymax, conf, conf, label]

    def box2df(self, box, id, frame, occ=0):
        return pd.DataFrame({'ID':[id], 'Frame':frame, 'xmin':box[0], 'ymin':box[1], 'xmax':box[2], 'ymax':box[3], 'Conf':box[4], 'Label':box[-1], 'Occlude':occ})
        
    def current_frame_analyze(self, boxes):
        ### boxes = (list) batch*sample*dimension
        ### sample = [xmin, ymin, xmax, ymax, confidence, confidence, class]
        boxes = boxes[0]
        nobj = np.array(boxes).shape[0]
        for i in range(nobj-1):
            for j in range(i+1,nobj):
                if boxes[i][-1] == boxes[j][-1]:
                    box1 = boxes[i][:4]
                    box2 = boxes[j][:4]
                    iou = self.cal_iou(box1, box2)
                    if iou>0.6:
                        del boxes[i]
                        nobj -= 1
                        break
        return [boxes]

    def data_association(self, target, last_boxes):
        max_iou = 0
        idx = -1
        for i in range(len(last_boxes[0])):
            last_box = last_boxes[0][i]
            this_iou = self.cal_iou(last_box, target)
            if (this_iou>self.thre) & (this_iou>max_iou):
                max_iou = this_iou
                idx = i
        return idx

    def cal_iou(self, box1, box2):
        # box = [xmin, ymin, xmax, ymax]
        area1 = np.zeros(self.res)
        area2 = np.zeros(self.res)
        area1[int(box1[0]):int(box1[2]), int(box1[1]):int(box1[3])] = 1
        area2[int(box2[0]):int(box2[2]), int(box2[1]):int(box2[3])] = 1
        return np.sum(np.logical_and(area1,area2)) / np.sum(np.logical_or(area1,area2))


if __name__ == '__main__':

    boxes0 = [[[0, 0, 2, 2, 0.9237443, 0.9237443, 1], 
              [8, 8, 10, 10, 0.9179137, 0.9179137, 2], 
              [3, 3, 5, 5, 0.9790077, 0.9790077, 3]]]
    boxes1 = [[[1, 1, 3, 3, 0.9237443, 0.9237443, 1], 
              [8, 7, 10, 9, 0.9179137, 0.9179137, 2]]]
    boxes2 = [[[1, 0, 3, 2, 0.9237443, 0.9237443, 1], 
              [7, 7, 9, 9, 0.9179137, 0.9179137, 2], 
              [4, 3, 6, 5, 0.9790077, 0.9790077, 3]]]
    BOX = [boxes0, boxes1, boxes2]


    # fig, ax = plt.subplots(1)
    # recs = []
    # c = ['r', 'b', 'y']

    st = Simple_tracker((20,20))
    for i in range(len(BOX)):
        st.obj_tracking(BOX[i], frame=i+1)

    st.plot_tracking(frame=0)
    #     print(st.tracking_obj)

    #     for box in BOX[i][0]:
    #         recs.append(pth.Rectangle((box[0],box[2]), width=(box[1]-box[0]), height=(box[3]-box[2]), color=c[i]))
    #     pc = PatchCollection(recs, match_original=True, alpha=0.2, edgecolor='b')
    #     ax.add_collection(pc)
    # ax.set_xlim([0,10])
    # ax.set_ylim([0,10])
    # ax.axis('equal')
    # plt.show()