"""
image.py: image dir processing code
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
import sys
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
        image_files: list[str],
        total_sample,
        batch_size,
        worker_index,
    ):
        super().__init__()
        self.processor_path = processor_path
        self.configuration_path = configuration_path
        self.image_files = image_files
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
            print(f"Preprocessor-{self.worker_index:02d} preprocess has exception: {error}")

    def _run(self):
        self.preprocess_function = load_module(self.processor_path, "preprocess")
        with open(self.configuration_path, "r") as fid:
            # self.config is a dictionary
            config = json.loads(fid.read())

        for img_file in self.image_files:
            if self.external_event.is_set():
                return

            cur_batch = []
            cur_metadata = []
            cur_filename = []

            try:
                frame = Frame(img_file)
                assert frame.height() > 0 and frame.width() > 0
            except BaseException:
                continue

            processed_input, metadata = self.preprocess_function(frame, config["preprocessing"])
            cur_batch.append(processed_input)
            cur_metadata.append(metadata)
            cur_filename.append(os.path.basename(img_file))
            if len(cur_batch) >= self.batch_size:
                # convert to numpy array
                cur_batch = np.concatenate(cur_batch, axis=0)
                while self.frames.qsize() > 100:
                    time.sleep(0.1)
                self.frames.put((cur_batch, cur_metadata, cur_filename))
                cur_batch = []
                cur_metadata = []
                cur_filename = []

        if len(cur_batch) > 0:
            cur_batch = np.concatenate(cur_batch, axis=0)
            self.frames.put((cur_batch, cur_metadata, cur_filename))

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
        output_dir: str,
        worker_index: int,
    ):
        super().__init__()
        self.processor_path = processor_path
        self.configuration_path = configuration_path
        self.worker_index = worker_index
        self.output_dir = output_dir
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

        binary_file = os.path.join(self.output_dir, f"{self.worker_index:09d}.bin")
        index_file = os.path.join(self.output_dir, f"{self.worker_index:09d}.idx")
        blob = BinaryBlob(binary_file=binary_file, index_file=index_file, mode="w")
        sample_idx = 0

        while True:
            if self.external_event.is_set() and self.data.empty():
                break

            if not self.data.empty():
                blob_idx, outputs, metadata, image_files = self.data.get()
                for output_, metadata_, file_ in zip(outputs, metadata, image_files):
                    output = postprocess_function(output_, metadata_, config["postprocessing"])
                    blob.write_index(sample_idx, (file_, output))
                    sample_idx += 1
            else:
                time.sleep(0.001)

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
        image_dir: str,
        file_per_worker: int,
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
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.file_per_worker = file_per_worker

    def run(self):
        try:
            return self.run_()
        except BaseException as error:
            if self.writer is not None:
                self.writer.request_close()
                self.writer.join()
            if self.preprocess is not None:
                self.preprocess.request_close()
                self.preprocess.join()

            print("====================================================")
            traceback.print_exc()
            print(f"Worker-{self.worker_index} has exception: {error}")
            print("====================================================")
            sys.exit(1)

    def run_(self):
        self.writer = None
        self.preprocess = None
        # compute total number of images that need to process
        total = 0
        image_files = [
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if not f.startswith(".")
        ]
        image_files.sort()
        start_idx = self.worker_index * self.file_per_worker
        stop_idx = min(len(image_files), (self.worker_index + 1) * self.file_per_worker)
        image_files = image_files[start_idx:stop_idx]

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
        self.preprocess = Preprocessor(
            processor_path=self.processor_path,
            configuration_path=self.configuration_path,
            image_files=image_files,
            batch_size=self.batch_size,
            worker_index=self.worker_index,
            total_sample=total,
        )
        self.preprocess.start()
        self.writer = DataWriter(
            processor_path=self.processor_path,
            configuration_path=self.configuration_path,
            output_dir=self.output_dir,
            worker_index=self.worker_index,
        )
        self.writer.start()
        count = 0

        # print every 2 %
        milestone = max(total // 50, 1)
        prog_bar = tqdm(total=total, desc=f"Worker-{self.worker_index:02d}", unit=" frame")

        while True:
            if self.preprocess.has_exception():
                self.writer.request_close()
                raise RuntimeError(f"Preprocessor-{self.worker_index:02d} has exception")

            if self.writer.has_exception():
                self.preprocess.request_close()
                raise RuntimeError(f"DataWriter-{self.worker_index:02d} has exception")

            if self.preprocess.has_finished() and not self.preprocess.has_frame():
                break

            if self.preprocess.has_frame():
                blob_idx, inputs, metadata, image_files = self.preprocess.get_frame()
                count += len(inputs)
                outputs = estimator.forward(inputs)
                # wait if queue is larger than 100
                while self.writer.queue_size() > 100:
                    time.sleep(0.1)
                self.writer.put_data((blob_idx, outputs, metadata, image_files))

                if count >= milestone:
                    prog_bar.update(count)
                    milestone = min(total, milestone + max(1, total // 50))

        self.writer.request_close()
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


def process_image_dir(
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
    image_files = [f for f in os.listdir(data_dir) if not f.startswith(".")]

    if nb_worker is None:
        if execution_provider == "CUDAExecutionProvider":
            nb_worker = get_gpu_count()
        else:
            nb_worker = os.cpu_count()

    nb_worker = min(nb_worker, len(image_files))
    file_per_worker = int(np.ceil(len(image_files) / nb_worker))

    processes = []
    start_time = time.time()
    for wrk_index in range(nb_worker):
        p = Processor(
            image_dir=data_dir,
            file_per_worker=file_per_worker,
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
