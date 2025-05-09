import cv2

def convert_avi_to_mp4(input_path: str, output_path: str):
    # Open the input AVI file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter for MP4 (H.264 encoding)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Conversion complete: {output_path}")

def normalize_box(box, width=640, height=480):
        """
        Convert gemini bounding box coordinates from meters to pixels.

        Args:
            box (List[float]): Bounding box in [ymin, xmin, ymax, xmax] format in meters.
            width (int): Image width.
            height (int): Image height.

        Returns:
            List[float]: Converted box in pixel coordinates.
        """
        ymin, xmin, ymax, xmax = box
        normalized_box = [
            int(xmin / 1000 * width),
            int(ymin / 1000 * height),
            int(xmax / 1000 * width),
            int(ymax / 1000 * height),
        ]
        print(f"Normalized box: {normalized_box}")
        return normalized_box
