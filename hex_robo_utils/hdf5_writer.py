#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-10-11
################################################################

import time
import threading
import multiprocessing
import os
import h5py
import numpy as np
from collections import deque


class HexHdf5Writer:

    def __init__(
        self,
        file_path: str,
        batch_size: int = 64,
    ):
        self.__file_path = file_path
        self.__hdf5_file = h5py.File(file_path, "w", libver='latest')
        self.__group_dict = {}
        self.__dataset_dict = {}
        self.__batch_size = batch_size

        self.__queue = deque()
        self.__stop_event = threading.Event()
        self.__writer_cnt = 0
        self.__writer_thread = None
        self.__writer_exc = None

    def __del__(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.__writer_thread and self.__writer_thread.is_alive():
            return
        self.__stop_event.clear()
        self.__writer_thread = threading.Thread(
            target=self.__writer_loop,
            daemon=True,
        )
        self.__writer_thread.start()

    def stop(self):
        self.__stop_event.set()
        if self.__writer_thread is not None and self.__writer_thread.is_alive(
        ):
            # Wait for writer thread to finish processing all queued data
            self.__writer_thread.join(timeout=30.0)  # Add timeout to avoid hanging
            if self.__writer_thread.is_alive():
                print("Warning: Writer thread did not finish in time, some data may be lost")
        # Ensure queue is empty (should be handled by writer_loop, but double-check)
        if len(self.__queue) > 0:
            print(f"Warning: {len(self.__queue)} items still in queue after writer thread stopped")
        if self.__hdf5_file is not None:
            try:
                self.__hdf5_file.flush()
                self.__hdf5_file.close()
            except Exception:
                pass
            self.__hdf5_file = None
        if self.__writer_exc:
            raise self.__writer_exc

    def get_state(self, dataset_name: str = "data"):
        """Get state (shape and dtype) for all groups."""
        state = {}
        for group_name in self.__group_dict.keys():
            try:
                dataset = self.__get_dataset_handle(group_name, dataset_name)
                state[group_name] = {
                    "shape": dataset.shape,
                    "dtype": dataset.dtype,
                }
            except (KeyError, AttributeError):
                # Skip groups that don't have the dataset
                continue
        return state

    def __get_dataset_handle(
        self,
        group_name: str,
        dataset_name: str = "data",
    ):
        dataset_key = f"{group_name}/{dataset_name}"
        if dataset_key not in self.__dataset_dict:
            if group_name not in self.__hdf5_file:
                raise KeyError(f"Group '{group_name}' not found in HDF5 file")
            if dataset_name not in self.__hdf5_file[group_name]:
                raise KeyError(
                    f"Dataset '{dataset_name}' not found in group '{group_name}'"
                )
            self.__dataset_dict[dataset_key] = self.__hdf5_file[group_name][
                dataset_name]
        return self.__dataset_dict[dataset_key]

    def __writer_loop(self):
        try:
            while not self.__stop_event.is_set():
                # Collect batch of items
                batch_items = []
                try:
                    # Try to collect up to batch_size items
                    for _ in range(self.__batch_size):
                        item = self.__queue.popleft()
                        batch_items.append(item)
                except IndexError:
                    # Queue is empty or not enough items
                    if len(batch_items) == 0:
                        time.sleep(1e-5)
                        continue

                # Write batch if we have items
                if batch_items:
                    self.__write_batch(batch_items)

            # Flush remaining items - process until queue is truly empty
            # Use a more robust approach: keep processing until queue is empty
            # with a safety check to avoid infinite loops
            max_iterations = 10000  # Safety limit
            iteration = 0
            while iteration < max_iterations:
                batch_items = []
                # Collect all available items (up to batch_size)
                try:
                    for _ in range(self.__batch_size):
                        item = self.__queue.popleft()
                        batch_items.append(item)
                except IndexError:
                    # Queue is empty, but write any collected items
                    if batch_items:
                        self.__write_batch(batch_items)
                    # Check if queue is truly empty
                    if len(self.__queue) == 0:
                        break
                    # Queue might have been populated, continue processing
                    if not batch_items:
                        time.sleep(1e-5)
                        iteration += 1
                        continue

                # Write batch if we have items
                if batch_items:
                    self.__write_batch(batch_items)
                    iteration = 0  # Reset counter when we process data
                else:
                    iteration += 1
        except Exception as e:
            self.__writer_exc = e
            raise

    def __write_batch(self, batch_items):
        """Write a batch of items, grouped by group_name for efficiency."""
        # Group items by group_name
        grouped_items = {}
        for item in batch_items:
            group = item[0]
            if group not in grouped_items:
                grouped_items[group] = []
            grouped_items[group].append(item)

        # Write each group's batch
        for group, items in grouped_items.items():
            self.__write_group_batch(group, items)

    def __write_group_batch(self, group_name, items):
        """Write a batch of items for a specific group."""
        dataset_key = f"{group_name}/data"
        get_ts_key = f"{group_name}/get_ts"
        sen_ts_key = f"{group_name}/sen_ts"
        ds = self.__dataset_dict[dataset_key]
        d_get = self.__dataset_dict[get_ts_key]
        d_sen = self.__dataset_dict[sen_ts_key]

        batch_size = len(items)
        n_old = ds.shape[0]
        n_new = n_old + batch_size

        # Resize all datasets once
        ds.resize((n_new, *ds.shape[1:]))
        d_get.resize((n_new, 1))
        d_sen.resize((n_new, 1))

        # Prepare batch arrays
        data_list = []
        gts_list = []
        sts_list = []

        for group, data, gts, sts in items:
            data_list.append(data)
            # Ensure gts and sts are scalars or 1-element arrays
            if isinstance(gts, np.ndarray):
                gts_val = gts.item() if gts.size == 1 else gts[0]
            else:
                gts_val = int(gts)
            if isinstance(sts, np.ndarray):
                sts_val = sts.item() if sts.size == 1 else sts[0]
            else:
                sts_val = int(sts)
            gts_list.append(gts_val)
            sts_list.append(sts_val)

        # Stack arrays for batch write
        data_batch = np.stack(data_list, axis=0)
        gts_batch = np.array(gts_list, dtype=np.int64).reshape(-1, 1)
        sts_batch = np.array(sts_list, dtype=np.int64).reshape(-1, 1)

        # Batch write
        ds[n_old:n_new, ...] = data_batch
        d_get[n_old:n_new, :] = gts_batch
        d_sen[n_old:n_new, :] = sts_batch

        self.__writer_cnt += batch_size

    def create_dataset(
        self,
        group_name: str,
        shape: tuple,
        dtype: np.dtype,
        chunk_num: int,
        max_num: int | None = None,
        compression=None,
    ):
        if group_name not in self.__group_dict:
            self.__group_dict[group_name] = self.__hdf5_file.create_group(
                group_name)

        dataset = self.__group_dict[group_name].create_dataset(
            "data",
            shape=(0, *shape),
            maxshape=(max_num, *shape),
            dtype=dtype,
            chunks=(chunk_num, *shape),
            compression=compression,
        )
        get_ts_set = self.__group_dict[group_name].create_dataset(
            "get_ts",
            shape=(0, 1),
            maxshape=(max_num, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )
        sen_ts_set = self.__group_dict[group_name].create_dataset(
            "sen_ts",
            shape=(0, 1),
            maxshape=(max_num, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )

        # Store dataset reference for easy access
        self.__dataset_dict[f"{group_name}/data"] = dataset
        self.__dataset_dict[f"{group_name}/get_ts"] = get_ts_set
        self.__dataset_dict[f"{group_name}/sen_ts"] = sen_ts_set

    def append_data(
        self,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray | int,
        sen_ts: np.ndarray | int,
    ):
        if isinstance(get_ts, int):
            get_ts = np.array([get_ts])
        if isinstance(sen_ts, int):
            sen_ts = np.array([sen_ts])
        item = (
            group_name,
            data,
            get_ts,
            sen_ts,
        )
        self.__queue.append(item)

    def append_batch_data(
        self,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray,
        sen_ts: np.ndarray,
    ):
        """Append batch data more efficiently by adding all items to queue at once."""
        batch_size = data.shape[0]
        # Ensure get_ts and sen_ts are properly shaped
        if get_ts.ndim == 0:
            get_ts = np.array([get_ts] * batch_size)
        elif get_ts.shape[0] != batch_size:
            raise ValueError(
                f"get_ts shape mismatch: expected {batch_size}, got {get_ts.shape[0]}"
            )

        if sen_ts.ndim == 0:
            sen_ts = np.array([sen_ts] * batch_size)
        elif sen_ts.shape[0] != batch_size:
            raise ValueError(
                f"sen_ts shape mismatch: expected {batch_size}, got {sen_ts.shape[0]}"
            )

        # Add all items to queue efficiently
        for i in range(batch_size):
            item = (
                group_name,
                data[i],
                get_ts[i] if isinstance(get_ts[i], np.ndarray) else np.array(
                    [get_ts[i]]),
                sen_ts[i] if isinstance(sen_ts[i], np.ndarray) else np.array(
                    [sen_ts[i]]),
            )
            self.__queue.append(item)

    def now_ns(self):
        return np.array([time.perf_counter_ns()])

    def hex_ts_to_ns(self, ts: dict):
        try:
            return np.array([ts["s"] * 1e9 + ts["ns"]])
        except Exception as e:
            print(f"hex_ts_to_ns failed: {e}")
            return np.array([np.inf])


def _writer_process_worker(file_path: str, batch_size: int,
                           data_queue: multiprocessing.Queue,
                           cmd_queue: multiprocessing.Queue,
                           stop_event: multiprocessing.Event):
    """Worker function running in a separate process for each writer."""
    import queue
    writer = HexHdf5Writer(file_path, batch_size)
    writer.start()

    def process_command(cmd):
        """Process a single command."""
        if cmd is None:
            return False  # Shutdown command
        cmd_type = cmd[0]
        if cmd_type == "create_dataset":
            group_name, shape, dtype, chunk_num, max_num = cmd[1:]
            writer.create_dataset(
                group_name=group_name,
                shape=shape,
                dtype=dtype,
                chunk_num=chunk_num,
                max_num=max_num,
                compression=None,
            )
        elif cmd_type == "get_state":
            dataset_name, result_queue = cmd[1:]
            state = writer.get_state(dataset_name)
            result_queue.put(state)
        return True

    try:
        # First, process all pending commands (especially create_dataset)
        # This ensures datasets are created before processing data
        while True:
            try:
                cmd = cmd_queue.get(timeout=0.1)
                if not process_command(cmd):
                    break  # Shutdown command
            except queue.Empty:
                break  # No more commands, proceed to data processing
            except Exception as e:
                print(f"Error processing command in writer process: {e}")

        # Now process data and commands in the main loop
        while not stop_event.is_set():
            # Check for commands with higher priority
            cmd_processed = False
            try:
                cmd = cmd_queue.get_nowait()
                if not process_command(cmd):
                    break  # Shutdown command
                cmd_processed = True
            except queue.Empty:
                pass  # No command available
            except Exception as e:
                print(f"Error processing command in writer process: {e}")

            # Process data from queue
            try:
                item = data_queue.get(
                    timeout=0.1 if not cmd_processed else 0.0)
                if item is None:  # Shutdown marker
                    break
                group_name, data, get_ts, sen_ts = item
                writer.append_data(group_name, data, get_ts, sen_ts)
            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                print(f"Error processing data in writer process: {e}")
                continue

        # Flush remaining data from data_queue
        while True:
            try:
                item = data_queue.get_nowait()
                if item is None:
                    break
                group_name, data, get_ts, sen_ts = item
                writer.append_data(group_name, data, get_ts, sen_ts)
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error flushing data in writer process: {e}")
                break

    finally:
        # Stop writer, which will flush its internal queue
        writer.stop()
        # Double-check that writer's internal queue is empty
        if hasattr(writer, '_HexHdf5Writer__queue') and len(writer._HexHdf5Writer__queue) > 0:
            print(f"Warning: Writer internal queue still has {len(writer._HexHdf5Writer__queue)} items after stop()")


class HexHdf5MultiWriter:

    def __init__(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        self.__base_dir = base_dir
        arm_path = f"{base_dir}/arms.h5"
        rgb_path = f"{base_dir}/rgb.h5"
        depth_path = f"{base_dir}/depth.h5"

        # Configuration for each writer type
        self.__writer_configs = {
            "robot": {
                "file_path": arm_path,
                "print_interval": 10_000,
                "batch_size": 1024,
            },
            "rgb": {
                "file_path": rgb_path,
                "print_interval": 300,
                "batch_size": 4,
            },
            "depth": {
                "file_path": depth_path,
                "print_interval": 300,
                "batch_size": 4,
            },
        }

        # Process management
        self.__processes: dict[str, multiprocessing.Process] = {}
        self.__data_queues: dict[str, multiprocessing.Queue] = {}
        self.__cmd_queues: dict[str, multiprocessing.Queue] = {}
        self.__stop_events: dict[str, multiprocessing.Event] = {}
        self.__manager = None
        self.__pending_commands: dict[str, list] = {}

        # For printing progress
        self.__writer_cnts: dict[str, int] = {}
        self.__print_nums: dict[str, int] = {}
        self.__print_intervals: dict[str, int] = {}
        self.__group_names: dict[str, set] = {}  # Track group names for each msg_type
        for msg_type, config in self.__writer_configs.items():
            self.__writer_cnts[msg_type] = 0
            self.__print_nums[msg_type] = 0
            self.__print_intervals[msg_type] = config["print_interval"]
            self.__group_names[msg_type] = set()

    def start(self):
        if self.__manager is None:
            self.__manager = multiprocessing.Manager()

        for msg_type, config in self.__writer_configs.items():
            if msg_type in self.__processes:
                continue  # Already started

            # Create queues and events
            data_queue = self.__manager.Queue()
            cmd_queue = self.__manager.Queue()
            stop_event = self.__manager.Event()

            self.__data_queues[msg_type] = data_queue
            self.__cmd_queues[msg_type] = cmd_queue
            self.__stop_events[msg_type] = stop_event

            # Start process
            p = multiprocessing.Process(
                target=_writer_process_worker,
                args=(
                    config["file_path"],
                    config["batch_size"],
                    data_queue,
                    cmd_queue,
                    stop_event,
                ),
            )
            p.start()
            self.__processes[msg_type] = p
            print(f"Started writer process for {msg_type}")

            # Execute pending commands (send before any data)
            if msg_type in self.__pending_commands:
                for cmd in self.__pending_commands[msg_type]:
                    cmd_queue.put(cmd)
                del self.__pending_commands[msg_type]

            # Give the process a moment to start and process initial commands
            time.sleep(0.01)

    def stop(self):
        # Signal all processes to stop
        for msg_type, stop_event in self.__stop_events.items():
            stop_event.set()
            # Send shutdown marker to data queue
            if msg_type in self.__data_queues:
                self.__data_queues[msg_type].put(None)
            # Send shutdown command to cmd queue
            if msg_type in self.__cmd_queues:
                self.__cmd_queues[msg_type].put(None)

        # Wait for all processes to finish
        print("Waiting for writer processes to finish...")
        for msg_type, process in self.__processes.items():
            if process.is_alive():
                process.join()
                if process.is_alive():
                    print(
                        f"Warning: {msg_type} writer process did not terminate, forcing kill"
                    )
                    process.terminate()
                    process.join()
            print(f"Stopped writer process for {msg_type}")

        self.__processes.clear()
        self.__data_queues.clear()
        self.__cmd_queues.clear()
        self.__stop_events.clear()

        if self.__manager is not None:
            self.__manager.shutdown()
            self.__manager = None

        # Wait a bit to ensure all file handles are closed
        time.sleep(0.1)

        # Generate summary after all processes stopped
        self.summary()

    def summary(self):
        """Print summary for all HDF5 files."""
        print("#" * 100)
        print(f"HDF5 Files Base Directory: {self.__base_dir}")
        print("#" * 100)

        for msg_type, config in self.__writer_configs.items():
            file_path = config["file_path"]
            if not os.path.exists(file_path):
                continue

            # Retry opening the file in case it's still being closed
            max_retries = 10
            retry_delay = 0.1
            f = None
            for attempt in range(max_retries):
                try:
                    f = h5py.File(file_path, "r")
                    break
                except (OSError, IOError) as e:
                    if "already open" in str(e) or "consistency flags" in str(
                            e):
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            print(
                                f"Warning: Could not open {file_path} after {max_retries} attempts: {e}"
                            )
                            continue
                    else:
                        raise

            if f is None:
                continue

            try:
                with f:
                    print("-" * 100)
                    print(f"File: {file_path}")
                    print("-" * 100)

                    for group_name in f.keys():
                        print(f"Group: {group_name}")
                        if 'get_ts' in f[group_name] and 'sen_ts' in f[
                                group_name]:
                            get_ts = f[group_name]['get_ts']
                            sen_ts = f[group_name]['sen_ts']

                            if len(get_ts) > 0 and len(sen_ts) > 0:
                                get_delta_s = (get_ts[-1][0] -
                                               get_ts[0][0]) * 1e-9
                                sen_delta_s = (sen_ts[-1][0] -
                                               sen_ts[0][0]) * 1e-9

                                print(f"  get_delta: {get_delta_s}s")
                                print(f"  sen_delta: {sen_delta_s}s")

                                if 'data' in f[group_name]:
                                    data_shape = f[group_name]['data'].shape
                                    dtype = f[group_name]['data'].dtype
                                    print(f"  Dataset: {group_name}/data")
                                    print(f"    Dtype: {dtype}")
                                    print(f"    Shape: {data_shape}")

                                    if np.fabs(get_delta_s) < 1e-9:
                                        print(
                                            f"    GetHz: N/A (all timestamps identical)"
                                        )
                                    else:
                                        print(
                                            f"    GetHz: {data_shape[0] / get_delta_s:.2f}"
                                        )
                                    if np.fabs(sen_delta_s) < 1e-9:
                                        print(
                                            f"    SenHz: N/A (all timestamps identical)"
                                        )
                                    else:
                                        print(
                                            f"    SenHz: {data_shape[0] / sen_delta_s:.2f}"
                                        )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                # Make sure file is closed even if there's an error
                try:
                    if f is not None:
                        f.close()
                except:
                    pass

        print("#" * 50)

    def _print_progress(self, msg_type: str):
        """Print progress information for a writer type."""
        if msg_type not in self.__cmd_queues:
            return
        
        try:
            state = self.get_state(msg_type, "data")
            if not state:
                return
            print("#" * 50)
            for group_name, info in state.items():
                shape = info.get("shape")
                if shape and len(shape) > 0:
                    length = shape[0]
                    print(f"{msg_type}/{group_name} len:{length}")
        except (ValueError, TimeoutError):
            # Writer not started or timeout, silently skip
            return
        except Exception:
            # Other errors, silently skip
            return

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_state(self, msg_type: str, dataset_name: str = "data"):
        """Get state (shape and dtype) for all groups from the writer process."""
        if msg_type not in self.__cmd_queues:
            raise ValueError(f"Writer for {msg_type} not started")
        import queue
        result_queue = self.__manager.Queue()
        cmd = ("get_state", dataset_name, result_queue)
        self.__cmd_queues[msg_type].put(cmd)
        try:
            return result_queue.get(timeout=5.0)
        except queue.Empty:
            raise TimeoutError(
                f"Timeout waiting for state from {msg_type} writer")

    def create_dataset(
        self,
        msg_type: str,
        group_name: str,
        shape: tuple,
        dtype: np.dtype,
        chunk_num: int,
        max_num: int | None = None,
    ):
        if msg_type not in self.__writer_configs:
            raise ValueError(f"Unknown writer type: {msg_type}")
        # Track group name for progress printing
        self.__group_names[msg_type].add(group_name)
        cmd = ("create_dataset", group_name, shape, dtype, chunk_num, max_num)
        if msg_type in self.__cmd_queues:
            # Process is started, send command directly
            self.__cmd_queues[msg_type].put(cmd)
        else:
            # Process not started yet, cache command
            if msg_type not in self.__pending_commands:
                self.__pending_commands[msg_type] = []
            self.__pending_commands[msg_type].append(cmd)

    def append_data(
        self,
        msg_type: str,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray | int,
        sen_ts: np.ndarray | int,
    ):
        if msg_type not in self.__data_queues:
            raise ValueError(f"Writer for {msg_type} not started")
        # Convert timestamps to int if needed
        if isinstance(get_ts, np.ndarray):
            if get_ts.size == 1:
                get_ts = int(get_ts.item())
            else:
                get_ts = int(get_ts[0])
        if isinstance(sen_ts, np.ndarray):
            if sen_ts.size == 1:
                sen_ts = int(sen_ts.item())
            else:
                sen_ts = int(sen_ts[0])
        item = (group_name, data, get_ts, sen_ts)
        self.__data_queues[msg_type].put(item)

        # Update counter and print progress
        self.__writer_cnts[msg_type] += 1
        cur_print_num = self.__writer_cnts[msg_type] // self.__print_intervals[
            msg_type]
        if cur_print_num > self.__print_nums[msg_type]:
            self.__print_nums[msg_type] = cur_print_num
            self._print_progress(msg_type)

    def append_batch_data(
        self,
        msg_type: str,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray,
        sen_ts: np.ndarray,
    ):
        """Append batch data by sending each item individually to the queue."""
        if msg_type not in self.__data_queues:
            raise ValueError(f"Writer for {msg_type} not started")
        batch_size = data.shape[0]

        # Ensure get_ts and sen_ts are properly shaped
        if get_ts.ndim == 0:
            get_ts = np.array([get_ts] * batch_size)
        elif get_ts.shape[0] != batch_size:
            raise ValueError(
                f"get_ts shape mismatch: expected {batch_size}, got {get_ts.shape[0]}"
            )

        if sen_ts.ndim == 0:
            sen_ts = np.array([sen_ts] * batch_size)
        elif sen_ts.shape[0] != batch_size:
            raise ValueError(
                f"sen_ts shape mismatch: expected {batch_size}, got {sen_ts.shape[0]}"
            )

        # Send each item to queue
        for i in range(batch_size):
            gts = int(
                get_ts[i]) if not isinstance(get_ts[i], np.ndarray) else int(
                    get_ts[i].item() if get_ts[i].size == 1 else get_ts[i][0])
            sts = int(
                sen_ts[i]) if not isinstance(sen_ts[i], np.ndarray) else int(
                    sen_ts[i].item() if sen_ts[i].size == 1 else sen_ts[i][0])
            item = (group_name, data[i], gts, sts)
            self.__data_queues[msg_type].put(item)

        # Update counter and print progress
        self.__writer_cnts[msg_type] += batch_size
        cur_print_num = self.__writer_cnts[msg_type] // self.__print_intervals[
            msg_type]
        if cur_print_num > self.__print_nums[msg_type]:
            self.__print_nums[msg_type] = cur_print_num
            self._print_progress(msg_type)

    def now_ns(self):
        return np.array([time.time_ns()])

    def hex_ts_to_ns(self, ts: dict):
        try:
            return np.array([ts["s"] * 1e9 + ts["ns"]])
        except Exception as e:
            print(f"hex_ts_to_ns failed: {e}")
            return np.array([np.inf])
