import subprocess
import sys
import os

def get_frame_rate(video_file):
    """Get the frame rate of the given video file using FFmpeg."""
    try:
        result = subprocess.run(["ffmpeg", "-i", video_file], stderr=subprocess.PIPE, text=True)
        output = result.stderr

        for line in output.split('\n'):
            if "fps" in line and "Video" in line:
                return int(line.split("kb/s, ")[1].split(" fps")[0])
    except Exception as e:
        print(f"Error processing file {video_file}: {e}")
        return None

def adjust_frame_rate(video_file, target_fps):
    """Adjust the frame rate of the given video file to the target frame rate."""
    output_file = f"adjusted_{os.path.basename(video_file)}"
    subprocess.run(["ffmpeg", "-i", video_file, "-filter:v", f"fps=fps={target_fps}", output_file])
    return output_file

def concatenate_videos(video_files, output_file="output.mp4"):
    """Concatenate a list of video files into a single video file."""
    with open("input.txt", "w") as f:
        for video in video_files:
            f.write(f"file '{video}'\n")

    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", output_file])
    os.remove("input.txt")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video1> <video2> ...")
        sys.exit(1)

    video_files = sys.argv[1:]
    highest_fps = 0

    # Identify the highest frame rate
    for video in video_files:
        fps = get_frame_rate(video)
        if fps:
            print(f"Frame rate of {video}: {fps} fps")
            if fps > highest_fps:
                highest_fps = fps

    if highest_fps == 0:
        print("Could not determine the highest frame rate.")
        sys.exit(1)

    print(f"Highest frame rate identified: {highest_fps} fps")

    # Adjust frame rates
    adjusted_videos = []
    for video in video_files:
        adjusted_video = adjust_frame_rate(video, highest_fps)
        adjusted_videos.append(adjusted_video)

    # Concatenate videos
    concatenate_videos(adjusted_videos)
    print("Videos have been concatenated into 'output.mp4'")

if __name__ == "__main__":
    main()

