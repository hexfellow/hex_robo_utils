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
    from hex_robo_utils.rerun_util import HexRerunParserUtil
except ImportError:
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hex_robo_utils.rerun_util import HexRerunParserUtil


def main():
    data_path = "multi_arm_rgbd/rerun_data.rrd"

    rerun_util = HexRerunParserUtil(data_path)
    print(rerun_util.summary())
    print(rerun_util.get_keys())

    # jnt_pos
    jnt_pos_dict = rerun_util.get_data("jnt/pos")
    print(jnt_pos_dict["ts_ns"].shape, jnt_pos_dict["data"].shape)
    print(
        f"jnt_pos ts_ns: {jnt_pos_dict['ts_ns'][0]}, jnt_pos data: {jnt_pos_dict['data'][0]}"
    )

    # rgb
    cam_rgb_dict = rerun_util.get_data("cam_0/rgb")
    print(cam_rgb_dict["ts_ns"].shape, cam_rgb_dict["data"].shape)

    # depth
    cam_depth_dict = rerun_util.get_data("cam_0/depth")
    print(cam_depth_dict["ts_ns"].shape, cam_depth_dict["data"].shape)
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
