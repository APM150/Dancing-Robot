from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

def combine_audio_video(video_file, audio_file, output_file):
    # Load video and audio files
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)

    # Calculate the number of loops required to match the audio duration
    video_duration = video.duration
    audio_duration = audio.duration
    num_loops = int(audio_duration // video_duration) + 1

    # Create a looped video with a duration equal to or greater than the audio duration
    looped_video = concatenate_videoclips([video] * num_loops)

    # Set the audio for the looped video
    video_with_audio = looped_video.set_audio(audio)

    # Write the result to the output file
    video_with_audio.write_videofile(output_file, codec='libx264', audio_codec='aac')

def trim_video(input_file, output_file, trim_duration=30):
    # Load the input video file
    video = VideoFileClip(input_file)

    # Trim the video to the specified duration
    trimmed_video = video.subclip(0, trim_duration)

    # Write the trimmed video to the output file
    trimmed_video.write_videofile(output_file, codec='libx264', audio_codec='aac')

# Usage example
video_file = '../runs/gait-conditioned-agility/pretrain-v1_L2reward/train_dance/data0-2/videos/deploy.mp4'
audio_file = '../custom_music/9i6bCWIdhBw.wav'
output_file = '../runs/gait-conditioned-agility/pretrain-v1_L2reward/train_dance/data0-2/videos/deploy_final.mp4'
combine_audio_video(video_file, audio_file, output_file)

long_file = output_file
short_file = '../runs/gait-conditioned-agility/pretrain-v1_L2reward/train_dance/data0-2/videos/deploy_final_short.mp4'
trim_video(long_file, short_file)
