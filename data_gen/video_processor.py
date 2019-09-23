import cv2
import numpy as np


class VideoProcessor:
    """ Video processor object that mainly fetches neighboring frames from a
    given frame_number. """
    def __init__(self, num_frames_before_after, enhance_clahe=False):
        # print "\nStarting processing videos and extracting frames and data..."
        self.frame_data = []
        self.num_frames_before_after = num_frames_before_after
        is_include_mid_in_frames = True
        if is_include_mid_in_frames: self.num_frames_before_after[1] += 1
        self.image_shape = (480, 640)  # TODO:
        self.enhance_clahe = enhance_clahe

    def enhance(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe_1 = clahe.apply(image[:, :, 0])
        img_clahe_2 = clahe.apply(image[:, :, 1])
        img_clahe_3 = clahe.apply(image[:, :, 2])
        img_clahe = cv2.merge((img_clahe_1, img_clahe_2, img_clahe_3))
        return img_clahe

    def preprocess_image(self, image):
        # if self.enhance_clahe:
        #     image = self.enhance(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        return image

    def get_video_frames(self, cap, frame_number):
        frame_names = []
        frame_images = np.zeros((self.num_frames_before_after[1] * 2 - 1,
                                 self.image_shape[0], self.image_shape[1]),
                                dtype=np.uint8)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success = frame_number + self.num_frames_before_after[
            0] > 0 and n_frames - frame_number > self.num_frames_before_after[1]
        if success:
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    frame_number + self.num_frames_before_after[0])
            for i in range(self.num_frames_before_after[0],
                           self.num_frames_before_after[1]):  # + 1):
                ret, frame = cap.read()
                if not ret:
                    ret, frame = cap.read()
                    if not ret:
                        return None, None
                        # raise Exception("VIDEO MIGHT BE CORRUPT!")
                frame = self.preprocess_image(frame)
                frame_name = str(frame_number + i) + '.png'
                frame_names.append(frame_name)
                # frame_images = np.concatenate((frame_images, frame[np.newaxis, :]), axis=0)
                frame_images[i + self.num_frames_before_after[0], ...] = frame

            # print("   > {0} frames added.".format(frame_images.shape[0]))
            return frame_images, frame_names
        else:
            print("Warning: image is skipped since frame number is {0}".format(
                frame_number))
            return None, None
