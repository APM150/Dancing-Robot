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

# Usage example
video_file = '../runs/gait-conditioned-agility/2023-05-05/train_dance/045109.997099/videos/deploy.mp4'
audio_file = '../custom_music/9i6bCWIdhBw.wav'
output_file = '../runs/gait-conditioned-agility/2023-05-05/train_dance/045109.997099/videos/deploy_final.mp4'
combine_audio_video(video_file, audio_file, output_file)
