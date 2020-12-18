from moviepy.editor import VideoFileClip

from helper import *
from lane_lines import *

# calibration images and camera matrix path
CALIBRATION_FILES_PATH = "./camera_cal"
# camera matric and distortion coefficient file
CAMERA_MATRIX_FILE = "camera_calibration.npy"

# output images
output_files_path = "./output_images/"


CHALLENGE_IMAGES_PATH = "./challenge_images"
challenge_images = glob.glob(CHALLENGE_IMAGES_PATH + "/frame*.jpg")


challenge_video = "./challenge_video.mp4"

# frames = get_frames_from_video(challenge_video, CHALLENGE_IMAGES_PATH + "/")
# status = create_video_from_frames(images, filename, codec="MJPG")

process_image = Pipeline(CALIBRATION_FILES_PATH + "/calibration*.jpg", 9, 6, debug=False, output_path=output_files_path)

def apply_images(image_files, debug=False):
    output_images = []
    output_image_names = []
    for img in image_files:
        # output file name
        path, filename = os.path.split(img)
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lane_image = process_image(image)
        # append result to output images
        output_images.append(lane_image)
        output_image_names.append(filename)
        if debug == True:
            plot_result([image, lane_image], ['image', 'lane'], rows=1, cols=2)

    status = write_images_to_folder(output_images, output_image_names, output_files_path)


def apply_video(input_video, output_video):    
    #clip1 = VideoFileClip(input_video).subclip(0,5) # subclip of the first 5 seconds
    clip1 = VideoFileClip(input_video)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output_video, audio=False)
    print("Video pipeline: SUCCESS!")


# test images
TEST_IMAGES_PATH = "./test_images"
test_images = glob.glob(TEST_IMAGES_PATH + "/test*.jpg")
apply_images(test_images)

# test videos
OUTPUT_VIDEOS_PATH = "./output_videos"
project_video = "./project_video.mp4"
project_video_ouput = OUTPUT_VIDEOS_PATH + "/project_video_output.mp4"
input_video = project_video
output_video = project_video_ouput
apply_video(input_video, output_video)
