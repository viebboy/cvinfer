"""
blob.py: image blob processing code
-----------------------------------


* Copyright: 2022 datsbot.com
* Authors: Dat Tran (hello@dats.bio)
* Date: 2023-10-28
* Version: 0.0.1


This is part of image_inferencer package

License
-------
Proprietary License

"""

from __future__ import annotations
import os
import numpy as np
import json
import traceback
from cvinfer.common import Frame, OnnxModel, BinaryBlob, load_module
import multiprocessing as MP
import threading
import time
from tqdm import tqdm
from queue import Queue

CTX = MP.get_context("spawn")


def get_gpu_count():
    result = os.popen("lspci | grep -i 'vga\\|3d\\|2d'").read()
    gpu_list = result.split('\n')
    gpu_count = len([gpu for gpu in gpu_list if gpu.strip()])
    return gpu_count


class Preprocessor(threading.Thread):
    def __init__(
        self,
        processor_path: str,
        configuration_path: str,
        image_blob_files: list[str],
        total_sample,
        batch_size,
        worker_index,
    ):
        super().__init__()
        self.processor_path = processor_path
        self.configuration_path = configuration_path
        self.image_blob_files = image_blob_files
        self.total_sample = total_sample
        self.batch_size = batch_size
        self.frames = Queue()
        self.internal_event = threading.Event()
        self.external_event = threading.Event()
        self.worker_index = worker_index

    def run(self):
        try:
            self._run()
        except BaseException as error:
            self.internal_event.set()
            traceback.print_exc()
            print(f"Preprocessor-{self.worker_index:02d} has exception: {error}")

    def _run(self):
        self.preprocess_function = load_module(self.processor_path, "preprocess")
        with open(self.configuration_path, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

        for blob_idx, (bin_file, idx_file) in enumerate(self.image_blob_files):
            cur_batch = []
            cur_metadata = []
            blob = BinaryBlob(binary_file=bin_file, index_file=idx_file, mode="r")
            for i in range(len(blob)):
                frame_data = blob.read_index(i)
                frame = Frame(frame_data)
                processed_input, metadata = self.preprocess_function(
                    frame, self.config["preprocessing"]
                )
                cur_batch.append(processed_input)
                cur_metadata.append(metadata)
                if len(cur_batch) >= self.batch_size:
                    # convert to numpy array
                    cur_batch = np.concatenate(cur_batch, axis=0)
                    while self.frames.qsize() > 100:
                        time.sleep(0.1)
                    self.frames.put((blob_idx, cur_batch, cur_metadata))
                    cur_batch = []
                    cur_metadata = []

            if len(cur_batch) > 0:
                cur_batch = np.concatenate(cur_batch, axis=0)
                self.frames.put((blob_idx, cur_batch, cur_metadata))

        self.external_event.set()
        print(f"Preprocessor-{self.worker_index:02d} is done")

    def has_frame(self):
        return not self.frames.empty()

    def get_frame(self):
        return self.frames.get()

    def has_finished(self):
        return self.external_event.is_set()

    def request_close(self):
        self.external_event.set()

    def has_exception(self):
        return self.internal_event.is_set()


class DataWriter(threading.Thread):
    def __init__(
        self,
        processor_path: str,
        configuration_path: str,
        nb_input_blob: str,
        output_dir: str,
        worker_index: int,
    ):
        super().__init__()
        self.processor_path = processor_path
        self.configuration_path = configuration_path
        self.worker_index = worker_index
        self.output_dir = output_dir
        self.nb_input_blob = nb_input_blob
        self.data = Queue()
        self.internal_event = threading.Event()
        self.external_event = threading.Event()

    def queue_size(self):
        return self.data.qsize()

    def run(self):
        try:
            self.run_()
        except BaseException as error:
            traceback.print_exc()
            print(f"DataWriter-{self.worker_index:02d} has exception: {error}")
            self.internal_event.set()

    def run_(self):
        postprocess_function = load_module(self.processor_path, "postprocess")
        with open(self.configuration_path, "r") as fid:
            # self.config is a dictionary
            config = json.loads(fid.read())

        self.blobs = []
        for idx in range(self.nb_input_blob):
            binary_file = os.path.join(self.output_dir, f"{idx:09d}.bin")
            index_file = os.path.join(self.output_dir, f"{idx:09d}.idx")
            self.blobs.append(BinaryBlob(binary_file=binary_file, index_file=index_file, mode="w"))

        self.blob_count = [0 for _ in self.blobs]
        while True:
            if self.external_event.is_set() and self.data.empty():
                break

            if not self.data.empty():
                blob_idx, outputs, metadata = self.data.get()
                for output_, metadata_ in zip(outputs, metadata):
                    output = postprocess_function(output_, metadata_, config["postprocessing"])
                    self.blobs[blob_idx].write_index(self.blob_count[blob_idx], output)
                    self.blob_count[blob_idx] += 1
            else:
                time.sleep(0.001)

        for blob in self.blobs:
            blob.close()

        print(f"DataWriter-{self.worker_index:02d} is done")

    def put_data(self, data):
        self.data.put(data)

    def request_close(self):
        self.external_event.set()

    def has_exception(self):
        return self.internal_event.is_set()


class Processor(CTX.Process):
    def __init__(
        self,
        image_blob_files: list[tuple[str, str]],
        output_dir: str,
        onnx_path: str,
        processor_path: str,
        configuration_path: str,
        execution_provider: str,
        worker_index: int,
        mem_limit: int,
        batch_size: int,
    ):
        super().__init__()
        self.onnx_path = onnx_path
        self.processor_path = processor_path
        self.configuration_path = configuration_path
        self.execution_provider = execution_provider
        self.worker_index = worker_index
        self.mem_limit = mem_limit
        self.batch_size = batch_size
        self.image_blob_files = image_blob_files
        self.output_dir = output_dir

    def run(self):
        try:
            return self.run_()
        except BaseException as error:
            print("====================================================")
            traceback.print_exc()
            print(f"worker-{self.worker_index} has exception: {error}")
            print("====================================================")

    def run_(self):
        # compute total number of images that need to process
        total = 0
        for bin_file, idx_file in self.image_blob_files:
            blob = BinaryBlob(binary_file=bin_file, index_file=idx_file, mode="r")
            total += len(blob)
            blob.close()

        estimator = OnnxModel(
            onnx_path=self.onnx_path,
            processor_path=self.processor_path,
            configuration_path=self.configuration_path,
            execution_provider=self.execution_provider,
            gpu_mem_limit=self.mem_limit
            if self.execution_provider == "CUDAExecutionProvider"
            else None,
            device_id=self.worker_index
            if self.execution_provider == "CUDAExecutionProvider"
            else None,
        )
        preprocess = Preprocessor(
            processor_path=self.processor_path,
            configuration_path=self.configuration_path,
            image_blob_files=self.image_blob_files,
            batch_size=self.batch_size,
            worker_index=self.worker_index,
            total_sample=total,
        )
        preprocess.start()
        writer = DataWriter(
            processor_path=self.processor_path,
            configuration_path=self.configuration_path,
            nb_input_blob=len(self.image_blob_files),
            output_dir=self.output_dir,
            worker_index=self.worker_index,
        )
        writer.start()
        count = 0

        # print every 2 %
        milestone = total // 50
        prog_bar = tqdm(total=total, desc=f"Worker-{self.worker_index:02d}", unit=" frame")

        while True:
            if preprocess.has_exception():
                writer.request_close()
                raise RuntimeError(f"Preprocessor-{self.worker_index:02d} has exception")

            if writer.has_exception():
                preprocess.request_close()
                raise RuntimeError(f"DataWriter-{self.worker_index:02d} has exception")

            if preprocess.has_finished() and not preprocess.has_frame():
                break

            if preprocess.has_frame():
                blob_idx, inputs, metadata = preprocess.get_frame()
                count += len(inputs)
                outputs = estimator.forward(inputs)
                # wait if queue is larger than 100
                while writer.queue_size() > 100:
                    time.sleep(0.1)
                writer.put_data((blob_idx, outputs, metadata))

                if count >= milestone:
                    prog_bar.update(count)
                    milestone = min(total, milestone + total // 50)

        writer.request_close()
        print(
            f"Worker-{self.worker_index:02d} processing is done. "
            f"Waiting for DataWriter-{self.worker_index:02d} to finalize"
        )


def monitor_and_terminate_processes(processes):
    try:
        error_process = None
        while True:
            all_exited = True
            for process in processes:
                if process.is_alive():
                    all_exited = False
                elif process.exitcode is not None and process.exitcode != 0:
                    error_process = process
                    break

            if error_process or all_exited:
                break
            time.sleep(0.5)

        if error_process:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError("Error happened in subprocess")
    except KeyboardInterrupt:
        for p in processes:
            if p.is_alive():
                p.terminate()

    except BaseException as e:
        traceback.print_exc()
        raise e


def process_image_blob(
    data_dir: str,
    output_dir: str,
    onnx_path: str,
    processor_path: str,
    configuration_path: str,
    mem_limit: int,
    batch_size: int,
    nb_worker: int,
    execution_provider: str,
):
    # find all bin files and idx files
    bin_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".bin")]
    bin_files.sort()

    idx_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".idx")]
    idx_files.sort()
    assert len(bin_files) == len(idx_files)

    if nb_worker is None:
        if execution_provider == "CUDAExecutionProvider":
            nb_worker = get_gpu_count()
        else:
            nb_worker = os.cpu_count()

    nb_worker = min(nb_worker, len(bin_files))
    file_per_worker = int(np.ceil(len(bin_files) / nb_worker))

    processes = []
    start_time = time.time()
    for wrk_index in range(nb_worker):
        bin_files_ = bin_files[wrk_index * file_per_worker : (wrk_index + 1) * file_per_worker]
        idx_files_ = idx_files[wrk_index * file_per_worker : (wrk_index + 1) * file_per_worker]

        image_files = [(bin_files_[i], idx_files_[i]) for i in range(len(bin_files_))]

        p = Processor(
            image_blob_files=image_files,
            output_dir=output_dir,
            onnx_path=onnx_path,
            processor_path=processor_path,
            configuration_path=configuration_path,
            execution_provider=execution_provider,
            worker_index=wrk_index,
            mem_limit=mem_limit,
            batch_size=batch_size,
        )
        p.start()
        processes.append(p)

    monitor_and_terminate_processes(processes)

    end_time = time.time()
    print(f"took {end_time - start_time} seconds")
