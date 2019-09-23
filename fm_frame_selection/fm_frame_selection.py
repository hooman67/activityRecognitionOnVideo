import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

sys.path.append('../')
from utils.utils import convert_bbox_coords_to_pixels, MaxHeap



class FmRecord:

    def __init__(
        self,
        frame,
        frameNumber,
        bboxes = None,
        filterdBboxes = None,
        hasWearArea = None,
        hasMatInside = None,
        hasBucket = None,
        hasTeethLine = None,
        numberOfDetectedTeeth = None,
        smallestToothLength = None,
        matInsideArea = None,
        imageWidth = None,
        imageHeight = None):


        self.frame = frame, 
        self.frameNumber = frameNumber
        self.bboxes = bboxes        
        self.hasWearArea = hasWearArea
        self.hasMatInside = hasMatInside
        self.hasBucket = hasBucket
        self.hasTeethLine = hasTeethLine
        self.numberOfDetectedTeeth = numberOfDetectedTeeth
        self.smallestToothLength = smallestToothLength
        self.matInsideArea = matInsideArea
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight



    # needed for the max heap
    def __lt__(self,other): return self.matInsideArea > other.matInsideArea
    def __eq__(self,other): return self.matInsideArea == other.matInsideArea
    def __str__(self): return str(self.matInsideArea)



    def count_and_identify_detected_objects(self, image_size):
        self.imageWidth, self.imageHeight = image_size, image_size

        self.smallestToothLength = 10000
        self.numberOfDetectedTeeth = 0

        #"labels":["Tooth", "Toothline", "BucketBB", "MatInside", "WearArea"],
        self.hasDetectionsOutsideImageBoundary = False
        self.hasWearArea = False
        self.hasMatInside = False
        self.hasBucket = False
        self.hasTeethLine = False
        self.matInsideArea = -1
        self.distanceBtw_xmins_of_matInsideAndBucket = -1
        self.distanceBtw_ymins_of_bucketAndTeethline = -1
        self.distanceBtw_xmins_of_bucketAndTeethline = -1
        self.distanceBtw_ymax_of_teethLine_and_ymin_of_matInside = -1
        
        for bbox in self.bboxes:
            if bbox.get_label() == 4: #WearArea
                self.hasWearArea = True

            
            elif bbox.get_label() == 3: #MatInside
                self.hasMatInside = True

                xmin_matInside, xmax_matInside, ymin_matInside, ymax_matInside = convert_bbox_coords_to_pixels(bbox, self.imageWidth, self.imageHeight)
                width_matInside = xmax_matInside - xmin_matInside
                height_matInside = ymax_matInside - ymin_matInside
                matInsideArea = width_matInside * height_matInside

                if xmax_matInside > self.imageWidth:
                    self.hasDetectionsOutsideImageBoundary = True


            elif bbox.get_label() == 2: #BucketBB
                self.hasBucket = True

                xmin_bucket, xmax_bucket, ymin_bucket, ymax_bucket = convert_bbox_coords_to_pixels(bbox, self.imageWidth, self.imageHeight)
                width_bucket = xmax_bucket - xmin_bucket
                height_bucket = ymax_bucket - ymin_bucket

                if xmax_bucket > self.imageWidth:
                    self.hasDetectionsOutsideImageBoundary = True

            
            elif bbox.get_label() == 1: #Toothline
                self.hasTeethLine = True

                xmin_teethLine, xmax_teethLine, ymin_teethLine, ymax_teethLine = convert_bboxCoordinates_to_pixels(bbox, self.imageWidth, self.imageHeight)
                width_teethLine = xmax_teethLine - xmin_teethLine
                height_teethLine = ymax_teethLine - ymin_teethLine

                if xmax_teethLine > self.imageWidth:
                    self.hasDetectionsOutsideImageBoundary = True

            
            elif bbox.get_label() == 0: #Tooth
                self.numberOfDetectedTeeth += 1

                xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(bbox, self.imageWidth, self.imageHeight)
                width = xmax - xmin
                height = ymax - ymin

                if height < self.smallestToothLength:
                    self.smallestToothLength = height


        if self.hasMatInside and self.hasBucket:
            self.distanceBtw_xmins_of_matInsideAndBucket = abs(xmin_matInside - xmin_bucket)

        if self.hasTeethLine and self.hasBucket:
            self.distanceBtw_ymins_of_bucketAndTeethline = abs(ymin_teethLine - ymin_bucket)
            self.distanceBtw_xmins_of_bucketAndTeethline = abs(xmin_teethLine - xmin_bucket)

        if self.hasTeethLine and self.hasMatInside:
            self.distanceBtw_ymax_of_teethLine_and_ymin_of_matInside = ymax_teethLine - ymin_matInside


class HsFmFrameSelector:

    def __init__(self, decisionMakingBufferLength, maxAllowableMinToothLength, outputPath, videoPath):
        self.decisionMakingBufferLength = decisionMakingBufferLength
        self.maxAllowableMinToothLength = maxAllowableMinToothLength
        self.outputPath = os.path.join(outputPath, videoPath.split('/')[-1][:-4])
        self.rejectedPath = os.path.join(self.outputPath, 'rejected')
        


        self.minDetected_numberOf_teeth = 0
        self.maxAllowable_distanceBtw_xminMatInside_and_xminBucket = 150 #150 final for hydraulics  #140 try5 #150 try4  
        self.maxAllowable_distanceBtw_yminTeethLine_and_yminBucket = 200 #200 final for hydraulics  #150 try5 #200 try4
        self.maxAllowable_distanceBtw_xminTeethLine_and_xminBucket = 125 #125 final for hydraulics
        self.minAllowable_signed_distance_matInside_below_teethLine = -1000 #1 final for hydraulics
        self.mustHaveTeethLine = False # True final for Hydraulics



        self.saveEveryFrameForDebug = True
        if self.saveEveryFrameForDebug:
            self.allFramesDir = os.path.join(self.outputPath, 'allFrames')
            if not os.path.exists(self.allFramesDir):
                os.makedirs(self.allFramesDir)


        #All buffered frames will be put here
        self.bufferedFramesDir = os.path.join(self.outputPath, 'bufferedFrames')
        if not os.path.exists(self.bufferedFramesDir):
            os.makedirs(self.bufferedFramesDir)



        #All final selected frames will be put here
        self.selectedFramesDir = os.path.join(self.outputPath, 'selectedFrames')
        if not os.path.exists(self.selectedFramesDir):
            os.makedirs(self.selectedFramesDir)

        

        self.rejected_missingBucketAndMatInsideDir = os.path.join(self.rejectedPath, 'accepted_hasMatInside')
        if not os.path.exists(self.rejected_missingBucketAndMatInsideDir):
            os.makedirs(self.rejected_missingBucketAndMatInsideDir)


        self.rejected_hasWearAreaDir = os.path.join(self.rejectedPath, 'rejected_hasWearArea')
        if not os.path.exists(self.rejected_hasWearAreaDir):
            os.makedirs(self.rejected_hasWearAreaDir)


        self.rejected_missingTeethlineDir = os.path.join(self.rejectedPath, 'rejected_missingTeethline')
        if not os.path.exists(self.rejected_missingTeethlineDir):
            os.makedirs(self.rejected_missingTeethlineDir)


        self.rejected_detectionOutOfImageBoundaryDir = os.path.join(self.rejectedPath, 'rejected_detectionOutOfImageBoundary')
        if not os.path.exists(self.rejected_detectionOutOfImageBoundaryDir):
            os.makedirs(self.rejected_detectionOutOfImageBoundaryDir)


        self.rejected_teethAre2BigDir = os.path.join(self.rejectedPath, 'rejected_teethAre2Big')
        if not os.path.exists(self.rejected_teethAre2BigDir):
            os.makedirs(self.rejected_teethAre2BigDir)


        self.rejected_overlap_teethLineAndBucketDir = os.path.join(self.rejectedPath, 'rejected_overlap_teethLineAndBucket')
        if not os.path.exists(self.rejected_overlap_teethLineAndBucketDir):
            os.makedirs(self.rejected_overlap_teethLineAndBucketDir)


        self.rejected_ovelap_matInsideAndTeethlineDir = os.path.join(self.rejectedPath, 'rejected_ovelap_matInsideAndTeethline')
        if not os.path.exists(self.rejected_ovelap_matInsideAndTeethlineDir):
            os.makedirs(self.rejected_ovelap_matInsideAndTeethlineDir)


        self.rejected_overlap_matInsideAndBucketDir = os.path.join(self.rejectedPath, 'rejected_overlap_matInsideAndBucket')
        if not os.path.exists(self.rejected_overlap_matInsideAndBucketDir):
            os.makedirs(self.rejected_overlap_matInsideAndBucketDir)



        self.receivedFrameNbs = []
        self.bufferedFrameNbs = []
        self.selectedFrameNbs = []

        self.rejected_missingBucketAndMatInside = []
        self.rejected_hasWearArea = []
        self.rejected_missingTeethline = []
        self.rejected_detectionOutOfImageBoundary = []
        self.rejected_teethAre2Big = []

        self.rejected_overlap_teethLineAndBucket = []
        self.rejected_ovelap_matInsideAndTeethline = []
        self.rejected_overlap_matInsideAndBucket = []


        
        self.buffer = MaxHeap()


    def update(self, fmRecord):

        self.receivedFrameNbs.append(int(fmRecord.frameNumber))

        if self.saveEveryFrameForDebug:              
            cv2.imwrite(os.path.join(self.allFramesDir ,"frame_" + str(fmRecord.frameNumber) + ".jpg")  , np.uint8(fmRecord.frame)[0,:,:,:]) 



        if(self.canBeBuffered(fmRecord)):
            self.buffer.heappush(fmRecord)

            self.bufferedFrameNbs.append(int(fmRecord.frameNumber))

            cv2.imwrite(os.path.join(self.bufferedFramesDir, 'bufferedFrame_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 



        if len(self.buffer) >= self.decisionMakingBufferLength:
            
            selectedRecord = self.buffer.heappop()
            
            self.selectedFrameNbs.append(int(selectedRecord.frameNumber))            

            cv2.imwrite(os.path.join(self.selectedFramesDir, 'selectedFrame_' + str(selectedRecord.frameNumber) + ".jpg"), np.uint8(selectedRecord.frame)[0,:,:,:]) 

            del self.buffer
            self.buffer = MaxHeap()


    def allRequiredObjectsWereDetected(self, fmRecord):
        if fmRecord.hasMatInside:# and fmRecord.hasBucket:

            #instead of doing this for frames missing matInside, do it for when we have the matInside
            self.rejected_missingBucketAndMatInside.append(int(fmRecord.frameNumber))
            cv2.imwrite(os.path.join(self.rejected_missingBucketAndMatInsideDir, 'accepted_hasMatInside_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 

            if not fmRecord.hasWearArea:

                if self.mustHaveTeethLine:
                    if fmRecord.hasTeethLine:
                        return True
                    else:
                        self.rejected_missingTeethline.append(int(fmRecord.frameNumber))
                        cv2.imwrite(os.path.join(self.rejected_missingTeethlineDir, 'rejected_missingTeethline_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
                        return False
                else:
                    return True    

            else:
                self.rejected_hasWearArea.append(int(fmRecord.frameNumber))
                cv2.imwrite(os.path.join(self.rejected_hasWearAreaDir, 'rejected_hasWearArea_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
                return False

        else:
            #self.rejected_missingBucketAndMatInside.append(int(fmRecord.frameNumber))
            #cv2.imwrite(os.path.join(self.rejected_missingBucketAndMatInsideDir, 'rejected_missingBucketAndMatInside_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
            return False


    def objectsHaveCorrectOverlap(self, fmRecord):
        if fmRecord.distanceBtw_ymins_of_bucketAndTeethline < self.maxAllowable_distanceBtw_yminTeethLine_and_yminBucket\
        and fmRecord.distanceBtw_xmins_of_bucketAndTeethline < self.maxAllowable_distanceBtw_xminTeethLine_and_xminBucket:

            if fmRecord.distanceBtw_ymax_of_teethLine_and_ymin_of_matInside > self.minAllowable_signed_distance_matInside_below_teethLine:

                if fmRecord.distanceBtw_xmins_of_matInsideAndBucket < self.maxAllowable_distanceBtw_xminMatInside_and_xminBucket:
                    return True
                else:
                    self.rejected_overlap_matInsideAndBucket.append(int(fmRecord.frameNumber))
                    cv2.imwrite(os.path.join(self.rejected_overlap_matInsideAndBucketDir, 'rejected_overlap_matInsideAndBucket_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
                    return False

            else:
                self.rejected_ovelap_matInsideAndTeethline.append(int(fmRecord.frameNumber))
                cv2.imwrite(os.path.join(self.rejected_ovelap_matInsideAndTeethlineDir, 'rejected_ovelap_matInsideAndTeethline_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
                return False

        else:
            self.rejected_overlap_teethLineAndBucket.append(int(fmRecord.frameNumber))
            cv2.imwrite(os.path.join(self.rejected_overlap_teethLineAndBucketDir, 'rejected_overlap_teethLineAndBucket_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
            return False


    def canBeBuffered(self, fmRecord):
        fmRecord.count_and_identify_detected_objects(image_size=640)

        if not self.allRequiredObjectsWereDetected(fmRecord):
            return False
        else:
            if fmRecord.hasDetectionsOutsideImageBoundary:
                self.rejected_detectionOutOfImageBoundary.append(int(fmRecord.frameNumber))
                cv2.imwrite(os.path.join(self.rejected_detectionOutOfImageBoundaryDir, 'rejected_detectionOutOfImageBoundary_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
                return False
            else:
                #if (fmRecord.numberOfDetectedTeeth > 0 and fmRecord.smallestToothLength > self.maxAllowableMinToothLength):
                #    self.rejected_teethAre2Big.append(int(fmRecord.frameNumber))
                #    cv2.imwrite(os.path.join(self.rejected_teethAre2BigDir, 'rejected_teethAre2Big_' + str(fmRecord.frameNumber) + ".jpg"), np.uint8(fmRecord.frame)[0,:,:,:]) 
                 #   return False
                #else:
                if not self.objectsHaveCorrectOverlap(fmRecord):
                    return False
                else:
                    return True


    def summerizeTheResults(self):
        self.receivedFrameNbs.sort()
        self.bufferedFrameNbs.sort()
        self.selectedFrameNbs.sort()

        self.rejected_missingBucketAndMatInside.sort()
        self.rejected_hasWearArea.sort()
        self.rejected_missingTeethline.sort()
        self.rejected_detectionOutOfImageBoundary.sort()
        self.rejected_teethAre2Big.sort()

        self.rejected_overlap_teethLineAndBucket.sort()
        self.rejected_ovelap_matInsideAndTeethline.sort()
        self.rejected_overlap_matInsideAndBucket.sort()


        plt.figure(figsize=(50,5))
        plt.xlabel('frameNumber(1 minute tick)')
        ax = plt.axes()
        loc = plticker.MultipleLocator(base=1800.0)
        ax.xaxis.set_major_locator(loc)
        ax.grid()
        

        plt.plot(self.receivedFrameNbs, len(self.receivedFrameNbs) * [0], label='received')
        plt.plot(self.bufferedFrameNbs, len(self.bufferedFrameNbs) * [0.1], 'bo', label='buffered')
        plt.plot(self.selectedFrameNbs, len(self.selectedFrameNbs) * [0.2], 'go', label='selected')

        plt.plot(self.rejected_missingBucketAndMatInside, len(self.rejected_missingBucketAndMatInside) * [0.6], 'x', color = 'chartreuse', label='containMatInside')
        plt.plot(self.rejected_hasWearArea, len(self.rejected_hasWearArea) * [0.3], 'gx', label='hasWearArea')
        plt.plot(self.rejected_missingTeethline, len(self.rejected_missingTeethline) * [0.3], 'rx', label='missingTeethline')
        plt.plot(self.rejected_detectionOutOfImageBoundary, len(self.rejected_detectionOutOfImageBoundary) * [0.4], 'cx', label='outOfBoundDetection')
        plt.plot(self.rejected_teethAre2Big, len(self.rejected_teethAre2Big) * [0.4], 'mx', label='teethAre2Big')
        plt.plot(self.rejected_overlap_teethLineAndBucket, len(self.rejected_overlap_teethLineAndBucket) * [0.5], 'yx', label='lol_teethLineAndBucket')
        plt.plot(self.rejected_ovelap_matInsideAndTeethline, len(self.rejected_ovelap_matInsideAndTeethline) * [0.5], 'kx', label='lol_matInsideAndTeethline')
        plt.plot(self.rejected_overlap_matInsideAndBucket, len(self.rejected_overlap_matInsideAndBucket) * [0.5],'bx', label='lol_matInsideAndBucket')

        plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)

        plt.savefig(os.path.join(self.outputPath, 'result_selectionTypes'))


        plt.figure(figsize=(40,10))

        resultsHistDic = {
        'received':len(self.receivedFrameNbs),
        'buffered':len(self.bufferedFrameNbs),
        'selected':len(self.selectedFrameNbs),
        'missingBucketOrMatInside':len(self.rejected_missingBucketAndMatInside),
        'hasWearArea':len(self.rejected_hasWearArea),
        'missingTeethline':len(self.rejected_missingTeethline),
        'outOfBoundDetection':len(self.rejected_detectionOutOfImageBoundary),
        'teethAre2Big':len(self.rejected_teethAre2Big),
        'lol_teethLineAndBucket':len(self.rejected_overlap_teethLineAndBucket),
        'lol_matInsideAndTeethline':len(self.rejected_ovelap_matInsideAndTeethline),
        'lol_matInsideAndBucket':len(self.rejected_overlap_matInsideAndBucket)
        }

        plt.bar(resultsHistDic.keys(), resultsHistDic.values(), color='g')

        plt.savefig(os.path.join(self.outputPath, 'result_selectionHistorgramAll'))




        plt.figure(figsize=(40,10))

        del resultsHistDic['received']
        del resultsHistDic['missingBucketOrMatInside']
        del resultsHistDic['buffered']

        plt.bar(resultsHistDic.keys(), resultsHistDic.values(), color='g')

        plt.savefig(os.path.join(self.outputPath, 'result_selectionHistorgramJustRejects'))
