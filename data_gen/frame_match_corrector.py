import numpy as np
import cv2
import matplotlib.pyplot as plt


class FrameMatchCorrector():
    """
    Fixes the bug that the h5['frames'] are not matched with the label. 
    correct_frame_match iterates over frames and tries to find a frame that matches the labeled frame.
    """

    def __init__(self, frames, image_path):
        self.frames = frames
        self.n_frames = frames.shape[0]
        self.image_path = image_path
        self.image_shape = (480, 640)  # TODO:

    def load_h5_into_ndarray(self, dset_name):
        ndarray = self.h5_file[dset_name].value
        return ndarray

    def find_correct_index(self, frame_data, frames, comparison_range, tol):
        index_range = (int((self.n_frames / 2 - comparison_range / 2)),
                       int((self.n_frames / 2 + comparison_range / 2)))
        for i, frame in enumerate(frames[index_range[0]:index_range[1], ...]):
            assert (frame_data.shape == frame.shape)
            is_equal = np.allclose(frame_data, frame, atol=tol)
            if is_equal:
                # plt.imshow(frame_data)
                # plt.show()
                # plt.imshow(frame)
                # plt.show()
                return index_range[0] + i

        return -1

    def save_h5(self):
        return

    def preprocess_image(self, image):
        # if self.args.enhance_clahe:
        #     image = self.enhance(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        return image

    def correct_frame_match(self):
        comparison_range = 12  # usually 5 frames wrong on one side
        new_n_frames = self.n_frames - comparison_range
        np_correct_frames = np.zeros((1, new_n_frames) + self.frames.shape[1:],
                                     dtype=np.float32)

        frame_data = cv2.imread(self.image_path)
        frame_data = self.preprocess_image(frame_data)
        index_mid = int((self.n_frames / 2))
        frame_mid = self.frames[index_mid, ...]
        tol = 10.
        is_equal = np.allclose(frame_data, frame_mid, atol=tol)
        if is_equal:
            np_correct_frames = np.concatenate((np_correct_frames, self.frames[int(
                comparison_range / 2):-int(comparison_range / 2), ...][np.newaxis, ...]),
                                               axis=0)
        else:
            frames = self.frames.copy()
            correct_frame_index = self.find_correct_index(frame_data, frames,
                                                          comparison_range, tol)
            if correct_frame_index == -1:  # not within comparison_range
                # print("Testing middle matches")
                # fig, ax = plt.subplots(1, 2, figsize=(12, 9))
                # mid = int(frames.shape[0] / 2)
                # ax[0].imshow(frames[mid, ...])
                # ax[1].imshow(frame_data)
                # plt.show()
                return None
            else:
                # print(correct_frame_index - index_mid, "frames away")
                remaining_min_number_of_frames = min(correct_frame_index,
                                                     self.n_frames - correct_frame_index)
                recovered_frames = frames[int(
                    (correct_frame_index - int(new_n_frames / 2))): int(
                    (correct_frame_index + int(new_n_frames / 2) + 1)), ...]
                assert (np.allclose(recovered_frames[int(len(recovered_frames) / 2)],
                                    frame_data, atol=tol))
                np_correct_frames = np.concatenate(
                    (np_correct_frames, recovered_frames[np.newaxis, ...]), axis=0)
                np_correct_frames = np_correct_frames[1:, ...]

        # print "Testing middle matches"
        # fig, ax = plt.subplots(1, 2)
        # mid = np_correct_frames.shape[1] / 2
        # ax[0].imshow(np_correct_frames[0, mid, ...])
        # ax[1].imshow(frame_data)
        # plt.show()

        return np_correct_frames
