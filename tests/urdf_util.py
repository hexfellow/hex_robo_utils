#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-11-28
################################################################

import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict


class HexDynUtils:

    class HexJointInfo:

        def __init__(self, name, joint_type, parent, child, axis, origin_xyz,
                     origin_rpy):
            self.name = name
            self.joint_type = joint_type  # 'revolute', 'prismatic', 'fixed'
            self.parent = parent  # parent link name
            self.child = child  # child link name
            self.axis = np.asarray(axis, float)
            self.origin_xyz = np.asarray(origin_xyz, float)
            self.origin_rpy = np.asarray(origin_rpy, float)

    class HexLinkInfo:

        def __init__(self, name, mass, com, inertia_3x3):
            self.name = name
            self.mass = float(mass)
            self.com = np.asarray(com, float)  # 3D COM position
            self.inertia = np.asarray(inertia_3x3, float)  # 3×3 inertia at COM

    def __init__(
            self,
            urdf_path: str,
            gravity: np.ndarray = np.array([0, 0, -9.81]),
    ):
        self.__links: Dict[str,
                           HexDynUtils.HexLinkInfo] = {}
        self.__joints: List[HexDynUtils.HexJointInfo] = []

        self.__parents: List[int] = []
        self.__child_index: List[int] = []
        self.__link_index_map: Dict[str, int] = {}

        self.__cartesian_inertia: List[np.ndarray] = []  # spatial inertia 6×6 per link
        self.__motion: List[np.ndarray] = []  # motion subspace 6×1 per joint
        self.__transform: List[np.ndarray] = [
        ]  # parent->joint constant transform (6×6)

        # gravity
        self.__gravity = gravity
        self.__load_from_urdf(urdf_path)

    # ---------------------------------------------------------
    #  URDF Loading
    # ---------------------------------------------------------
    def __load_from_urdf(self, path: str):
        """
        Parse URDF file and fill:
            self.links
            self.joints
            self.parents
            self.child_index
            self.link_index_map

        Then call:
            self.build_spatial_inertia_and_Xtree()
        """
        tree = ET.parse(path)
        root = tree.getroot()

        # Parse links
        for link_elem in root.findall('link'):
            link_name = link_elem.get('name')
            
            # Parse inertial properties
            inertial = link_elem.find('inertial')
            if inertial is not None:
                # Parse origin (COM position)
                origin = inertial.find('origin')
                if origin is not None:
                    xyz_str = origin.get('xyz', '0 0 0')
                    com = [float(x) for x in xyz_str.split()]
                else:
                    com = [0.0, 0.0, 0.0]
                
                # Parse mass
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value', '0'))
                else:
                    mass = 0.0
                
                # Parse inertia matrix
                inertia_elem = inertial.find('inertia')
                if inertia_elem is not None:
                    ixx = float(inertia_elem.get('ixx', '0'))
                    iyy = float(inertia_elem.get('iyy', '0'))
                    izz = float(inertia_elem.get('izz', '0'))
                    ixy = float(inertia_elem.get('ixy', '0'))
                    ixz = float(inertia_elem.get('ixz', '0'))
                    iyz = float(inertia_elem.get('iyz', '0'))
                    
                    # Build 3x3 inertia matrix
                    inertia_3x3 = np.array([
                        [ixx, ixy, ixz],
                        [ixy, iyy, iyz],
                        [ixz, iyz, izz]
                    ])
                else:
                    inertia_3x3 = np.zeros((3, 3))
            else:
                # Default values if no inertial
                com = [0.0, 0.0, 0.0]
                mass = 0.0
                inertia_3x3 = np.zeros((3, 3))
            
            # Create link info
            link_info = self.HexLinkInfo(link_name, mass, com, inertia_3x3)
            self.__links[link_name] = link_info

        # Parse joints
        for joint_elem in root.findall('joint'):
            joint_name = joint_elem.get('name')
            joint_type = joint_elem.get('type', 'fixed')
            
            # Parse origin
            origin = joint_elem.find('origin')
            if origin is not None:
                xyz_str = origin.get('xyz', '0 0 0')
                rpy_str = origin.get('rpy', '0 0 0')
                origin_xyz = [float(x) for x in xyz_str.split()]
                origin_rpy = [float(x) for x in rpy_str.split()]
            else:
                origin_xyz = [0.0, 0.0, 0.0]
                origin_rpy = [0.0, 0.0, 0.0]
            
            # Parse parent and child
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            if parent_elem is not None:
                parent_link = parent_elem.get('link')
            else:
                parent_link = None
            if child_elem is not None:
                child_link = child_elem.get('link')
            else:
                child_link = None
            
            # Parse axis
            axis_elem = joint_elem.find('axis')
            if axis_elem is not None:
                xyz_str = axis_elem.get('xyz', '0 0 1')
                axis = [float(x) for x in xyz_str.split()]
            else:
                axis = [0.0, 0.0, 1.0]
            
            # Create joint info
            joint_info = self.HexJointInfo(
                joint_name, joint_type, parent_link, child_link,
                axis, origin_xyz, origin_rpy
            )
            self.__joints.append(joint_info)

        # Build topology
        # First, create link_index_map (base_link is index 0, then others)
        link_names = list(self.__links.keys())
        # Ensure base_link is first if it exists
        if 'base_link' in link_names:
            link_names.remove('base_link')
            link_names.insert(0, 'base_link')
        
        for idx, link_name in enumerate(link_names):
            self.__link_index_map[link_name] = idx
        
        # Build child_index and parents
        for joint_idx, joint in enumerate(self.__joints):
            # child_index: joint -> child link index
            if joint.child in self.__link_index_map:
                child_idx = self.__link_index_map[joint.child]
                self.__child_index.append(child_idx)
            else:
                self.__child_index.append(-1)
            
            # parents: find parent joint index for this joint
            # The parent joint is the one whose child is this joint's parent link
            # If parent is base_link (index 0), then parent_joint_idx = -1
            parent_joint_idx = -1
            if joint.parent in self.__link_index_map:
                parent_link_idx = self.__link_index_map[joint.parent]
                # If parent is not base_link, find the joint that has this parent link as its child
                if parent_link_idx > 0:
                    for pj_idx, pj in enumerate(self.__joints):
                        if pj_idx < joint_idx and pj.child == joint.parent:
                            parent_joint_idx = pj_idx
                            break
            self.__parents.append(parent_joint_idx)

        # Call build_spatial_inertia_and_Xtree()
        self.build_spatial_inertia_and_Xtree()

    # ---------------------------------------------------------
    #  Building dynamic parameters: I, Xtree, S
    # ---------------------------------------------------------
    def build_spatial_inertia_and_Xtree(self):
        """
        After links/joints/topology are known, compute:
            - spatial inertia I[i]
            - motion subspace S[i]
            - Xtree[i]  (spatial transform from parent link → joint frame)
        """
        # TODO: build I[i] via mcI()
        # TODO: build S[i] from joint type & axis
        # TODO: build Xtree[i] from origin xyz/rpy
        raise NotImplementedError

    # ---------------------------------------------------------
    #  Joint transform for RNEA/CRBA
    # ---------------------------------------------------------
    def jcalc(self, joint_type: str, q_i: float, axis: np.ndarray):
        """
        Compute:
            XJ : joint transform (6×6)
            S  : motion subspace (6×1)

        The simplified version:
            For revolute:
                rotation about axis by q_i
            For prismatic:
                translation along axis * q_i
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #  RNEA (inverse dynamics)
    # ---------------------------------------------------------
    def rnea(self, q: np.ndarray, dq: np.ndarray,
             ddq: np.ndarray) -> np.ndarray:
        """
        Compute joint torque:
            tau = M(q) ddq + C(q, dq) + g(q)
        via recursive Newton–Euler.

        Requires:
            parents[]
            S[]
            I[]
            Xtree[]
            jcalc()

        Returns tau (n,)
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #  Mass matrix M(q)
    # ---------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Minimal implementation:
            For each basis vector e_i, compute:
                M[:, i] = rnea(q, 0, e_i)
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #  Gravity vector g(q)
    # ---------------------------------------------------------
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        g(q) = rnea(q, dq=0, ddq=0)
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #  Coriolis/centrifugal C(q, dq)
    # ---------------------------------------------------------
    def bias_force(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        C(q, dq) @ dq = rnea(q, dq, 0) - g(q)
        """
        raise NotImplementedError

    def coriolis_matrix(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Build full C(q,dq) by evaluating bias_force with dq = e_i.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #  Jacobian J(q) for a given link/frame
    # ---------------------------------------------------------
    def frame_jacobian(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        """
        Compute geometric Jacobian (6*n).

        Requires:
            parents[]
            S[]
            Xtree[]
            joint transforms (via jcalc)
            link_index_map
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #  Jacobian derivative Jdot(q,dq)
    # ---------------------------------------------------------
    def frame_jacobian_dot(self, q: np.ndarray, dq: np.ndarray,
                           frame_name: str) -> np.ndarray:
        """
        Minimal version using numerical differentiation:
            Jdot ≈ (J(q + eps*dq) - J(q)) / eps
        """
        eps = 1e-6
        J0 = self.frame_jacobian(q, frame_name)
        J1 = self.frame_jacobian(q + eps * dq, frame_name)
        return (J1 - J0) / eps
