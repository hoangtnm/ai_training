import os
import sys

import cv2
import torch

from utils import get_device
from utils import get_metadata
from utils import get_net
from utils import get_prediction_class
from utils import preprocess_image

if __name__ == '__main__':
    device = get_device()

    # Training dataset metadata
    _, class_names, class_to_idx = get_metadata(sys.argv[1])
    num_classes = len(class_names)
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    # Net initialization
    net = get_net(classes=num_classes)
    checkpoint_dict = torch.load(os.path.join('checkpoint', 'checkpoint.pth'),
                                 map_location=device)
    net.load_state_dict(checkpoint_dict['model_state_dict'])
    net.eval()
    net.to(device)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prediction
        frame_tensor = preprocess_image(frame_rgb, mode='val')
        frame_tensor = frame_tensor.to(device)
        prediction = net(frame_tensor)
        result = get_prediction_class(prediction, idx_to_class)
        print(f'Result: {result}')

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
