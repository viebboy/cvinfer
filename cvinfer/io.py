"""
io.py: IO utility
-----------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

License
-------
Apache License 2.0


"""

import os
import cv2
from loguru import logger
from cvinfer.common import Frame
from queue import Queue
import time
import threading


@logger.catch
def read_video(path, output_queue, input_queue, skip_frame_frequency, max_queue_size=100):
    """
    read a video as a collection of Frames and put into a queue
    :param path: (str) path to video
    :param output_queue: (queue.Queue) the queue for readout
    :param input_queue: (queue.Queue) if this queue is non-empty, the function terminates
    :param skip_frame_frequency: (int) if positive, skip a given number of frame after every readout
    """

    try:
        handle = cv2.VideoCapture(path)
        width = handle.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = handle.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = handle.get(cv2.CAP_PROP_FPS)
        frame_count = handle.get(cv2.CAP_PROP_FRAME_COUNT)

        # put metadata
        output_queue.put(
            {
                "width": int(width),
                "height": int(height),
                "fps": fps,
                "frame_count": frame_count,
            }
        )

        # put frames
        skip_count = skip_frame_frequency
        while True:
            if not input_queue.empty():
                # if receive any signal from parent
                logger.debug("input queue is non-empty, stop from reading video now")
                handle.release()
                output_queue.put(None)
                break
            elif output_queue.qsize() <= max_queue_size:
                # read frame
                is_valid, frame = handle.read()
                if is_valid:
                    # if read successfully
                    if skip_frame_frequency == skip_count:
                        # enough frame has been skipped
                        # note that frame generated by cv2 is in BGR order
                        # thus requires reverting
                        output_queue.put(Frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                        skip_count = 0
                    else:
                        skip_count += 1
                else:
                    # reading not successful, terminate
                    logger.debug("end of the stream")
                    handle.release()
                    output_queue.put(None)
                    break
            else:
                time.sleep(0.001)

    except Exception as error:
        # clean up
        output_queue.put(None)
        raise error


class VideoReader:
    """
    interface to read from video file
    call reader.next() to get the next video frame (an instance of Frame)
    call reader.close() to close the reader
    """

    def __init__(
        self,
        path,
        use_threading,
        skip_frame_frequency=0,
        max_queue_size=100,
        max_elapsed_time=20,
    ):
        if not hasattr(self, "_path"):
            # if the base implementation, a.k.a reading from file, assert
            assert os.path.exists(path)
            logger.debug(f"reading from video file: {path}")

        logger.debug(f"use threading: {use_threading}")

        self.use_threading = use_threading
        if use_threading:
            self._frame_queue = Queue()
            self._event_queue = Queue()
            self._capture_thread = threading.Thread(
                target=read_video,
                args=(path, self._frame_queue, self._event_queue, skip_frame_frequency),
            )
            # start the thread
            self._capture_thread.start()

            # wait until the metadata is read
            start_time = time.time()
            while self._frame_queue.empty():
                time.sleep(0.01)
                if time.time() - start_time > max_elapsed_time:
                    logger.warning(
                        "reaching the maximum elapsed time but receive no video metadata from IO thread"
                    )
                    self._terminate_thread()
                    raise RuntimeError()
                break
            self._metadata = self._frame_queue.get()
        else:
            self._video_handle = cv2.VideoCapture(path)
            width = self._video_handle.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self._video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self._video_handle.get(cv2.CAP_PROP_FPS)
            frame_count = self._video_handle.get(cv2.CAP_PROP_FRAME_COUNT)

            self._metadata = {
                "width": int(width),
                "height": int(height),
                "fps": fps,
                "frame_count": frame_count,
            }
            self._skip_frame_frequency = skip_frame_frequency
            self._skip_count = skip_frame_frequency

        self._max_elapsed_time = max_elapsed_time
        self._is_terminated = False

    def close(self):
        if not self._is_terminated:
            logger.debug("terminating now")
            if self.use_threading:
                self._event_queue.put("terminate")
                self._capture_thread.join()
            else:
                self._video_handle.release()
            self._is_terminated = True

    def metadata(self):
        return self._metadata

    @logger.catch
    def next(self):
        try:
            return self._next()
        except KeyboardInterrupt:
            logger.debug("keyboard interrupt, closing reader now")
            self.close()
            return
        except Exception as error:
            raise error

    def _next(self):
        if self._is_terminated:
            return

        start_time = time.time()
        if self.use_threading:
            # if use threading, wait within some period to get a frame
            # terminate if elapsed time exceeds the period
            while self._frame_queue.empty():
                time.sleep(0.001)
                if time.time() - start_time > self._max_elapsed_time:
                    logger.warning(
                        "reaching the maximum elapsed time but receive no video metadata from IO thread"
                    )
                    self.close()
                    raise RuntimeError()
                break
            frame = self._frame_queue.get()
            if frame is None:
                self.close()
            return frame
        else:
            # require looping because of frame skipping feature
            while True:
                is_valid, frame = self._video_handle.read()
                if is_valid:
                    # if read successfully
                    if self._skip_frame_frequency == self._skip_count:
                        # enough frame has been skipped
                        # note that frame generated by cv2 is in BGR order
                        # thus requires reverting
                        frame = Frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        self._skip_count = 0
                        return frame
                    else:
                        self._skip_count += 1
                else:
                    # reading not successful, terminate
                    self.close()
                    break


class RTSPReader(VideoReader):
    """
    reader for rtsp stream
    support reader.next() and reader.close()
    """

    def __init__(
        self,
        path,
        use_threading,
        skip_frame_frequency=0,
        max_queue_size=100,
        max_elapsed_time=20,
    ):
        assert path.startswith("rtsp") or path.startswith("RTSP")
        logger.debug(f"reading from RTSP stream: {path}")
        self._path = path

        super().__init__(
            path,
            use_threading,
            skip_frame_frequency,
            max_queue_size,
            max_elapsed_time,
        )

    def flush(self):
        """
        flush all frames in the queue and return the latest one
        """
        if self.use_threading:
            # if use threading, wait within some period to get a frame
            # terminate if elapsed time exceeds the period
            frame = None
            while not self._frame_queue.empty():
                frame = self._frame_queue.get()
            if frame is None:
                frame = self.next()
            return frame
        else:
            return self.next()


class WebcamReader(VideoReader):
    """
    interface to get frames from webcam
    support reader.next() and reader.close()
    """

    def __init__(
        self,
        cam_id: int,
        use_threading: bool,
        skip_frame_frequency=0,
        max_queue_size=100,
        max_elapsed_time=20,
    ):
        assert isinstance(cam_id, int)
        assert cam_id >= 0
        logger.debug(f"reading from Webcam stream with ID: {cam_id}")
        self._path = cam_id

        super().__init__(
            cam_id,
            use_threading,
            skip_frame_frequency,
            max_queue_size,
            max_elapsed_time,
        )

    def flush(self):
        """
        flush all frames in the queue and return the latest one
        """
        if self.use_threading:
            # if use threading, wait within some period to get a frame
            # terminate if elapsed time exceeds the period
            frame = None
            while not self._frame_queue.empty():
                frame = self._frame_queue.get()
            if frame is None:
                frame = self.next()
            return frame
        else:
            return self.next()


class VideoWriter:
    def __init__(self, video_file, fps):
        self._video_file = video_file
        self._metadata = {"fps": fps, "frame_count": 0}
        self._cv_writer = None

    def path(self):
        return self._video_file

    def metadata(self):
        return self._metadata

    def write(self, frame):
        if self._metadata["frame_count"] == 0 and self._video_file is not None:
            height = frame.height()
            width = frame.width()
            self._cv_writer = cv2.VideoWriter(
                self._video_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._metadata["fps"],
                (width, height),
            )

        if self._cv_writer is not None:
            # note that opencv has default BGR format so we need to flip
            # because Frame data is in RGB
            self._cv_writer.write(frame.data()[:, :, ::-1])
        else:
            cv2.namedWindow("cvinfer.io.VideoWriter", cv2.WINDOW_NORMAL)
            cv2.imshow("cvinfer.io.VideoWriter", frame.bgr())
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                raise KeyboardInterrupt()
        self._metadata["frame_count"] += 1

    def close(self):
        if self._cv_writer is not None:
            self._cv_writer.release()
