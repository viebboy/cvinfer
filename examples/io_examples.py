"""
io_examples.py: example usage of cvinfer.io
-------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

License
-------
Apache License 2.0


"""

from cvinfer.io import VideoReader, VideoWriter
import urllib.request
import os

# let's download a sample video first
remote_path = 'https://download.samplelib.com/mp4/sample-5s.mp4'
local_path = 'test_video.mp4'

if not os.path.exists(local_path):
    urllib.request.urlretrieve(remote_path, local_path)


"""
Below is a typical boilerplate of VideoReader, RTSPReader, WebcamReader

"""

reader = VideoReader(
    path=local_path,
    use_threading=True, # if True, load the video in another thread
    skip_frame_frequency=1, # skip 1 frame between every returned frames
)

"""
we could also create a reader object from an RTSP stream, or from the webcam stream

reader = RTSPReader(path=rtsp_url,.....)

reader = WebcamReader(cam_id=.....) # typically cam_id is 0 if machine has only 1 webcam


methods of VideoReader, RTSPReader or WebcamReader are the same

"""

# we will also create an instance of VideoWriter to save the processed video
writer = VideoWriter('processed_video.mp4', fps=60) # increase the FPS to increase playback speed

# access the metadata, which is a dictionary with "height", "width", "fps" as
# keys
metadata = reader.metadata()
print(f'video height: {metadata["height"]}')
print(f'video width: {metadata["width"]}')
print(f'video fps: {metadata["fps"]}')


# iteratively process the frames
# it's a good idea to put the processing logics in a try...except

try:
    while True:
        frame = reader.next() # this returns an instance of cvinfer.common.Frame
        if frame is None: # end of the video
            break

        # process this frame
        # because frame is an instance of cvinfer.common.Frame
        # we could leverage its methods like resizing, horizontal flipping,
        # color jiterring etc

        # resize
        # output of resize is the resized frame and the resize ratio
        frame, _ = frame.resize(
            new_width=300,
            new_height=300,
            keep_ratio=True, # resize by keeping aspect ratio
            pad_constant=255, # if keeping aspect ratio, we need to pad, so here specify the pad value
            interpolation='cubic',
        )

        # horizontal flip and random color jittering
        frame = frame.horizontal_flip().jitter_color()

        # then write this processed frame to writer
        writer.write(frame)

    # close the reader and writer
    reader.close()
    writer.close()

except Exception as error:
    # close the reader and writer
    reader.close()
    writer.close()
    raise error
