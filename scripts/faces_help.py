import cv2 as cv
import numpy as np
import face_alignment
    from skimage import io
import skimage.transform as trans
import os
import matplotlib.pyplot as plt


def get_affine(src, img_size=224):
    # src is the three points eye, eye, mouth
    # The coordinates below are for 224 image
    ratio = img_size / 224
    dst = np.array([[87,  59],
                    [137,  59],
                    [112, 120]], dtype=np.float32)
    dst = dst * ratio
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    return M


def affine_align_img(img, M, crop_size=224):
    warped = cv.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
    return warped

def create_frames():
    filename = '/Users/michaelko/Downloads/me_1008.mov'
    # filename = '/Users/michaelko/Downloads/actioncliptrain00648.avi'

    cap = cv.VideoCapture(filename)
    init = 0
    old_embeddings = None
    diff_list = []
    ret, frame = cap.read()
    frame_num = 0
    fps = cap.get(cv.CAP_PROP_FPS)
    while cap.isOpened() and ret:
        number = 5
        img_name = '/Users/michaelko/Downloads/me_movie1/' + f'{frame_num:06d}.jpg'
        # img_name = '/Users/michaelko/Downloads/frames/' + f'{frame_num:06d}.jpg'
        cv.imwrite(img_name, frame)
        frame_num = frame_num + 1
        ret, frame = cap.read()


def get_eyes_mouths(landmark):
    three_points = np.zeros((3, 2))
    three_points[0] = landmark[36:42].mean(0)
    three_points[1] = landmark[42:48].mean(0)
    three_points[2] = landmark[60:68].mean(0)
    return three_points


def affine_align_3landmarks(landmarks, M):
    new_landmarks = np.concatenate([landmarks, np.ones((3, 1))], 1)
    affined_landmarks = np.matmul(new_landmarks, M.transpose())
    return affined_landmarks


def get_eyes_mouths(landmark):
    three_points = np.zeros((3, 2))
    three_points[0] = landmark[36:42].mean(0)
    three_points[1] = landmark[42:48].mean(0)
    three_points[2] = landmark[60:68].mean(0)
    return three_points


def get_mouth_bias(three_points, img_size=224):
    ratio = img_size / 224
    bias = ratio*np.array([112, 120]) - three_points[2]
    return bias


def get_landmarks_from_folder(fa, folder_path):
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f[-3:] == 'jpg']

    onlyfiles = sorted(onlyfiles)

    preds = {}
    for f in onlyfiles:
        # if f != '000004.jpg':
        #     continue
        file_path = os.path.join(folder_path, f)
        input = io.imread(file_path)
        image_preds = fa.get_landmarks(input)
        preds[f] = image_preds
        print('Done ', f, len(image_preds))

    return preds


def prepare_image(fa, img_in, p_bias=None):
    """
    Given a face image, crop it so that only one face is found
    :param file_name:
    :return:
    """
    input = io.imread(img_in)
    img_size = input.shape[0]
    pred_points = np.array(fa.get_landmarks(input))

    if pred_points is None or len(pred_points.shape) != 3:
        print('preprocessing failed')
        return False, None, None

    num_faces, size, _ = pred_points.shape
    if num_faces == 1 and size == 68:
        three_points = get_eyes_mouths(pred_points[0])
    else:
        print('preprocessing failed')
        return False, None, None

    avg_points = three_points
    M = get_affine(avg_points, img_size)
    affined_3landmarks = affine_align_3landmarks(three_points, M)
    bias = get_mouth_bias(affined_3landmarks, img_size)

    if p_bias is None:
        bias = bias
    else:
        bias = p_bias * 0.2 + bias * 0.8

    M_i = M.copy()
    M_i[:, 2] = M[:, 2] + bias
    img = cv.imread(img_in)
    wrapped = affine_align_img(img, M_i, crop_size=img_size)

    # img_save_path = os.path.join(out_folder, img_pth)
    # img_save_path = os.path.join(folder_save_path, img_pth.split('/')[-1])
    wrapped = cv.resize(wrapped, (224, 224))
    # cv.imwrite(img_save_path, wrapped)

    return True, wrapped, bias


def preprocess_movie_folder(in_folder, out_folder):
    """
    Given a folder with frames of a movie, preprocess the frames
    :param folder_path:
    :param out_folder:
    :return:
    """
    onlyfiles = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f)) and f[-3:] == 'jpg']
    onlyfiles = sorted(onlyfiles)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

    p_bias = None
    for f in onlyfiles:
        file_path = os.path.join(in_folder, f)
        # Crop and wrap
        status, wrapped, p_bias = prepare_image(fa, file_path, p_bias)
        if not status:
            print('preprocessing failed')
            return
        # Save file to output folder
        img_save_path = os.path.join(out_folder, f)
        cv.imwrite(img_save_path, wrapped)

        print('Done ', f)


def find_faces(folder_path, out_folder):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    # preds = fa.get_landmarks_from_directory(folder_path)
    preds = get_landmarks_from_folder(fa, folder_path)

    sumpoints = 0
    three_points_list = []

    # plt.imshow(input)
    # for detection in preds:
    #     plt.scatter(detection[:, 0], detection[:, 1], 2)

    cnt = 0
    img_size = 224
    for img in preds.keys():
        if cnt == 0:
            img_size = cv.imread(os.path.join(folder_path, img)).shape[0]
            cnt = cnt + 1
        pred_points = np.array(preds[img])
        if pred_points is None or len(pred_points.shape) != 3:
            print('preprocessing failed')
            return False
        else:
            num_faces, size, _ = pred_points.shape
            if num_faces == 1 and size == 68:

                three_points = get_eyes_mouths(pred_points[0])
                sumpoints += three_points
                three_points_list.append(three_points)
            else:

                print('preprocessing failed')
                return False
    avg_points = sumpoints / len(preds)
    M = get_affine(avg_points, img_size)
    p_bias = None
    for i, img_pth in enumerate(preds.keys()):
        three_points = three_points_list[i]
        affined_3landmarks = affine_align_3landmarks(three_points, M)
        bias = get_mouth_bias(affined_3landmarks, img_size)
        if p_bias is None:
            bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M_i = M.copy()
        M_i[:, 2] = M[:, 2] + bias
        img = cv.imread(os.path.join(folder_path, img_pth))
        wrapped = affine_align_img(img, M_i, crop_size=img_size)
        # img_pth = f'{i:06}' + '.jpg'
        img_save_path = os.path.join(out_folder, img_pth)
        # img_save_path = os.path.join(folder_save_path, img_pth.split('/')[-1])
        wrapped = cv.resize(wrapped, (224, 224))
        cv.imwrite(img_save_path, wrapped)
    print('cropped files saved at {}'.format(out_folder))


def make_images_smaller(folder_path):
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f[-3:] == 'jpg']

    for f in onlyfiles:
        file_path = os.path.join(folder_path, f)
        img = cv.imread(file_path)
        img = img[420:-420, :, :]
        cv.imwrite(file_path, img)

def copy_images(folder_path1, folder_path2):
    onlyfiles = [f for f in os.listdir(folder_path1) if os.path.isfile(os.path.join(folder_path1, f)) and f[-3:] == 'jpg']

    for f in onlyfiles:
        file_path = os.path.join(folder_path1, f)
        img = cv.imread(file_path)
        img = cv.resize(img, (540, 540))
        file_path = os.path.join(folder_path2, f)
        cv.imwrite(file_path, img)


if __name__ == '__main__':

    # create_frames()
    # make_images_smaller('/Users/michaelko/Downloads/me_movie1')
    # find_faces('/Users/michaelko/Downloads/me_movie2')

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    prepare_image(fa, '/Users/michaelko/Downloads/alan2--1-.jpeg')

    # find_faces('/Users/michaelko/Downloads/me_movie2', '/Users/michaelko/Downloads/me_movie3')
    # copy_images('/Users/michaelko/Downloads/me_movie1', '/Users/michaelko/Downloads/me_movie2')
