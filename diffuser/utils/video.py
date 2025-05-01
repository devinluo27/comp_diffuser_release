import os, torch
import numpy as np
import skvideo.io
import imageio

def save_imgs_to_mp4(imgs, save_path, fps=24, n_repeat_first=0, n_rep_list=None):
    """
    Saves a list of NumPy imgs to an MP4 video file using imageio.

    Parameters:
    ----------
    imgs : list of np.ndarray
        List of imgs to be saved as video frames. Each image should be a NumPy array.
    save_path : str
        The filename for the output video file (e.g., 'output_video.mp4').
    fps : int, optional
        Frames per second for the output video. Default is 24.

    Returns:
    -------
    None

    Notes:
    -----
    - All imgs must have the same dimensions and number of channels.
    - imgs should be in uint8 format. If not, they will be converted.
    - If imgs are grayscale (2D arrays), they will be converted to RGB.
    """
    if torch.is_tensor(imgs):
        assert imgs.ndim == 4
        if imgs.shape[1] == 3:
            imgs = imgs.permute(0,2,3,1).numpy()
    elif type(imgs) == list:
        imgs = np.array(imgs)
    
    if n_repeat_first > 0:
        print(f'before: {imgs.shape=}')
        imgs = np.concatenate(   [imgs[0:1],] * n_repeat_first + [ imgs, ], axis=0  )
        print(f'after: {imgs.shape=}')
    elif n_rep_list is not None:
        num_fr = len(imgs)
        assert len(n_rep_list) == num_fr
        imgs_ori = imgs
        imgs = []
        ## repeat the corresponding frame n times
        for i_im in range(num_fr):
            imgs.extend( [imgs_ori[i_im],] * n_rep_list[i_im] )

    # Validate inputs
    # if not imgs:
        # raise ValueError("The list of images is empty.")

    if not isinstance(save_path, str):
        raise TypeError("Output filename must be a string.")

    ## Check ffmpeg availability
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # from diffuser.utils.eval_utils import suppress_warnings, suppress_stdout

    # print('Begin save video.')

    with imageio.get_writer(save_path, fps=fps) as writer:
        for idx, img in enumerate(imgs):
            # Convert image to uint8 if necessary
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 1) if img.dtype in [np.float32, np.float64] else img
                img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img.astype(np.uint8)
            
            # Convert grayscale to RGB
            if img.ndim == 2:
                img = np.stack((img,)*3, axis=-1)
            
            # Validate image dimensions
            if img.ndim != 3 or img.shape[2] not in [1, 3, 4]:
                raise ValueError(f"Invalid image shape at index {idx}: expected 2D or 3D array with 1, 3, or 4 channels.")

            # if True:
            # with suppress_stdout():
                # Append frame to video
            # print(f'{idx=}')
            writer.append_data(img)

    import diffuser.utils as utils
    utils.print_color(f'[save_imgs_to_mp4] to {save_path}')
    
    return 

def pad_videos_to_maxlen(vid_list, rep_c_scale=1.0):
    """
    vid_list: a list of np 4d (T,h,w,3)
    rep_c_scale: scaling factor for the last img so we know it it terminated.
    """
    assert vid_list[0].dtype == np.uint8

    vid_lens = [len(v) for v in vid_list]
    max_len = max(vid_lens)
    outs = []
    for i_v in range(len(vid_list)):
        tmp_len = len(vid_list[i_v])
        last_img = (vid_list[i_v][-1] * rep_c_scale).astype(np.uint8)
        ## [ (t,h,w,3) ,...,(1,h,w,3) ]
        tmp_v = [ vid_list[i_v], ] + [last_img[None,],] * (max_len - tmp_len)
        tmp_v = np.concatenate( tmp_v, axis=0 )
        outs.append( tmp_v )
    ## an array video of same len
    return np.array(outs)

def pad_videos_side(vid_list, pad_len):
    """
    given a list/array of videos, pad the four sides of each video
    """
    vid_list = np.pad(vid_list, 
                      ((0,0), (0,0), (pad_len,)*2, (pad_len,)*2, (0,0)), constant_values=255)
    return vid_list


# def pad_videos_side_color_v2(vid_list, pad_len, ):
def videos_loco_mark_color(vid_list, is_marks, ):
    """
    given a list/array of videos, pad the four sides of each video
    """
    # pad_vid_list = []
    vid_list = np.copy(vid_list)
    n_vids = len(vid_list)
    assert len(is_marks) == len(vid_list)
    for i_v in range(n_vids):
        vid = vid_list[i_v]
        img_h = vid.shape[1] ## t h w 3
        mark_size = img_h // 6
        mark_size = min(30, mark_size)
        if is_marks[i_v]:
            vid[:, :mark_size, :mark_size, :] = np.array([255,0,0]) ## green
        else:
            vid[:, :mark_size, :mark_size, :] = np.array([0,255,0]) ## green
    
    return vid_list


def pad_imgs_side(imgs, pad_len, pad_v):
    """
    given a list/array of imgs (4d), (N,H,W,3), pad the four sides of each img
    """
    imgs = np.pad(imgs, 
                      ( (0,0), (pad_len,)*2, (pad_len,)*2, (0,0) ), constant_values=pad_v)
    return imgs


def _make_dir(filename):
    """
    given path to a specific file, and make dir for that file
    """
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def load_mp4_video(video_file):
    """
    Extract frames from a video file and return them as a NumPy array.

    Parameters:
    video_file (str): Path to the video file.

    Returns:
    numpy.ndarray: An array of frames extracted from the video.
    """
    reader = imageio.get_reader(video_file, 'ffmpeg')
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    frames_array = np.stack(frames, axis=0)
    return frames_array



def save_video(filename, video_frames, fps=60, video_format='mp4'):
    assert False, 'not used'
    assert fps == int(fps), fps
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )

def save_videos(filename, *video_frames, axis=1, **kwargs):
    assert False, 'not used'
    ## video_frame : [ N x H x W x C ]
    video_frames = np.concatenate(video_frames, axis=axis)
    save_video(filename, video_frames, **kwargs)
