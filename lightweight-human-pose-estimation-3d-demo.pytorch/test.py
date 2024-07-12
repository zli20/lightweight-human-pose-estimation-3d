import cv2
from modules.input_reader import ImageReader
import numpy as np
import os
from modules.legacy_pose_extractor import extract_poses
from modules.parse_poses import parse_poses
import json


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def preprocess():
    path = [r"C:\Users\26522\Desktop\R.jpg"]
    # img = cv2.imread(path)
    frame_provider = ImageReader(path)
    for frame in frame_provider:
        input_scale = 256 / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % 8)]

        # cv2.imshow(" ", scaled_img)
        # cv2.waitKey(0)

        n, c, h, w = 1, 3, 256, 448
        img_mean = np.array([128, 128, 128], dtype=np.float32)
        img_scale = np.float32(1 / 255)
        if h != scaled_img.shape[0] or w != scaled_img.shape[1]:
            # 正常化图像
            normalized_img = (scaled_img - img_mean) * img_scale

            resized_img = cv2.resize(normalized_img, (w, h), interpolation=cv2.INTER_LINEAR)

            normalized_img_transposed = np.transpose(resized_img, (2, 0, 1))

            data = np.expand_dims(normalized_img_transposed, axis=0)

            print(scaled_img.shape)

        print(scaled_img.shape)


def postprocess():
    features = np.random.rand(57, 32, 56)
    heatmaps = np.random.rand(19, 32, 56)
    paf_map = np.random.rand(38, 32, 56)
    upsample_ratio = 4
    found_poses = extract_poses(heatmaps[0:-1], paf_map, upsample_ratio)[0]
    pass


def net_test():
    from modules.inference_engine_pytorch import InferenceEnginePyTorch
    net = InferenceEnginePyTorch(r"C:\Users\26522\Downloads\human-pose-estimation-3d.pth", "CPU")

    from modules.draw import Plotter3d, draw_poses
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    path = [r"C:\Users\26522\Desktop\R.jpg"]
    frame_provider = ImageReader(path)
    for frame in frame_provider:
        current_time = cv2.getTickCount()
        if frame is None:
            break
        input_scale = 256 / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:,
                     0:scaled_img.shape[1] - (scaled_img.shape[1] % 8)]  # better to pad, but cut out for demo
        scaled_img = cv2.resize(scaled_img, dsize=(448, 256), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow(" ", scaled_img)
        # cv2.waitKey(0)
        fx = np.float32(0.8 * frame.shape[1])
        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride=8, fx=fx)

        edges = []
        file_path = None
        if file_path is None:
            file_path = os.path.join('data', 'extrinsics.json')
        with open(file_path, 'r') as f:
            extrinsics = json.load(f)
        R = np.array(extrinsics['R'], dtype=np.float32)
        t = np.array(extrinsics['t'], dtype=np.float32)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]

            xx = -z
            yy = x
            zz = -y
            ax.scatter(xx, yy, zz, marker='o')

            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d,edges)
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        # # 设置图形属性
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_title('3D Poses Visualization')
        # 显示图形
        plt.show()


        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        mean_time = 0
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imshow('ICV 3D Human Pose Estimation', frame)

        delay = 1
        esc_code = 27
        p_code = 112
        space_code = 32
        mean_time = 0
        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1
        if 1 :  # allow to rotate 3D canvas while on pause
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1

    pass


if __name__ == '__main__':
    # preprocess()
    # postprocess()
    net_test()
    pass
