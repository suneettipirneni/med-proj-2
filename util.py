import torch
from monai.inferers import sliding_window_inference

from monai.transforms import Activations, AsDiscrete, Compose

from constants import IMAGE_SIZE


# define inference method
def inference(model, input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=IMAGE_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    with torch.cuda.amp.autocast():
      return _compute(input)

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])