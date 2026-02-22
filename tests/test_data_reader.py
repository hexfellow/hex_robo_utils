#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-21
################################################################

import os
import cv2

try:
    from hex_robo_utils import HexPandasReader, hdf5_to_pd, rerun_to_pd
except ImportError:
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hex_robo_utils import HexPandasReader, hdf5_to_pd, rerun_to_pd

TEST_TYPE = "rerun"


def main():
    raw_path, pd_path = None, None
    if TEST_TYPE == "hdf5":
        raw_path = "multi_arm_rgbd/hdf5_data"
        pd_path = f"{raw_path}_pd"
        hdf5_to_pd(raw_path, pd_path)
    elif TEST_TYPE == "rerun":
        raw_path = "multi_arm_rgbd/rerun_data"
        pd_path = f"{raw_path}_pd"
        rerun_to_pd(raw_path, pd_path)
    else:
        raise ValueError(f"Invalid test type: {TEST_TYPE}")

    reader = HexPandasReader(pd_path)
    print(reader.summary())
    print(reader.get_keys())

    # jnt_pos
    jnt_pos_dict = reader.get_data("arm_0/pos")
    print(jnt_pos_dict["sen_ts"].shape, jnt_pos_dict["data"].shape)
    print(
        f"jnt_pos sen_ts: {jnt_pos_dict['sen_ts'][0]}, jnt_pos data: {jnt_pos_dict['data'][0]}"
    )

    # rgb
    cam_rgb_dict = reader.get_data("cam_0/rgb")
    print(cam_rgb_dict["sen_ts"].shape, cam_rgb_dict["data"].shape)

    # depth
    cam_depth_dict = reader.get_data("cam_0/depth")
    print(cam_depth_dict["sen_ts"].shape, cam_depth_dict["data"].shape)
    depth = cam_depth_dict["data"][0]
    print(f"depth dtype: {depth.dtype}, depth shape: {depth.shape}")
    depth_u8 = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    depth_cmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    # show rgb and depth
    cv2.imshow("cam_rgb", cam_rgb_dict["data"][0])
    cv2.imshow("cam_depth", depth_cmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
