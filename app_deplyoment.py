import streamlit as st
import cv2
import torch
import config
from config import args_setting
from torch.utils.data import Dataset
from model import generate_model
from torchvision import transforms
from PIL import Image
import numpy as np


class RoadSequenceDatasetList(Dataset):

    def __init__(self, tensors):
        self.data_tensor = tensors
        self.dataset_size = len(self.data_tensor)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        sample = {'data': self.data_tensor}
        return sample


def output_result(model, test_loader, device):
    model.eval()
    feature_dic = []
    with torch.no_grad():
        for sample_batched in test_loader:
            data = sample_batched['data'].to(device)
            for param in model.parameters():
                param.grad = None
            output, feature = model(data)
            feature_dic.append(feature)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            img = Image.fromarray(img.astype(np.uint8))

            data = torch.squeeze(data).cpu().numpy()
            if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
                data = np.transpose(data, [1, 2, 0]) * 255
            else:
                data = np.transpose(data, [1, 2, 0]) * 255
            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = (img.getpixel((i, j)))
                    if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            return data



if __name__ == '__main__':
    print('[INFO] Starting System...')
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model and weights
    print('[INFO] Importing pretrained model..')
    model = generate_model(args)
    pretrained_dict = torch.load(config.pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # real time lane and curves detection from a video or camera
    print('[INFO] Starting live detection ')
    capture = cv2.VideoCapture('lanes_clip.mp4')

    out = cv2.VideoWriter('lanes_clip_detection.avi', cv2.VideoWriter_fourcc(*"MJPG"), 15, (512, 256))

    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    # turn image into floatTensor
    op_transforms = transforms.Compose([transforms.ToTensor()])
    i = 0
    while run:
        i = i+1
        if i % 5 == 0:
            _, frame = capture.read()
            # converting the frame from opencv format to pillow format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_data = Image.fromarray(image)
            # converting the image to have 3 channels <RGB>
            if image_data.mode != 'RGB':
                image_data = image_data.convert('RGB')
            # resizing the input
            new_size_image = image_data.resize((256, 128))
            # unsqueeze the data to be in 3 dimension
            image_data = [torch.unsqueeze(op_transforms(new_size_image), dim=0)]
            image_data = torch.cat(image_data, 0)

            # load data for batches, num_workers for multiprocess
            test_loader = torch.utils.data.DataLoader(RoadSequenceDatasetList(tensors=image_data),
                                                     batch_size=args.test_batch_size, shuffle=False, num_workers=1)
            # calling the output function to give us the prediction and resize then convert it to opencv format
            outresult = output_result(model, test_loader, device).resize((512, 256))
            cv_image = np.array(outresult)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            # show the result in the screen and save it
            out.write(cv2.resize(cv_image, (512, 256)))
            # cv2.imshow("result", cv_image)
            FRAME_WINDOW.image(cv_image)
            cv2.waitKey(1)

    out.release()
    cv2.destroyAllWindows()
