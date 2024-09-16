
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch

# Create a new blank image (32x32) with a black background
txtimage = Image.new("RGB", (32, 32), (0, 0, 0))

# Font size and loading a font (Arial in this case)
#font_size = 15  # Reduce font size to fit in 32x32 image
#font = ImageFont.truetype("arial.ttf", font_size)

# Create a dummy tensor patch (random for this example)
img_tensor_patch = torch.rand(3, 32, 32)  # 3-channel image

# Convert the tensor to a PIL image
to_pil = transforms.ToPILImage()
pil_image = to_pil(img_tensor_patch)

# Initialize a drawing context for the PIL image
draw = ImageDraw.Draw(pil_image)

# Define the maximum value (for demo purposes)
max_per_prototype = 99  # Replace this with your real value

# Draw text in the middle of the image with white color
draw.text((16, 16), str(0.6),
          #font=font,
          anchor='mm', fill="red")

# Convert the updated PIL image back to a tensor
txttensor = transforms.ToTensor()(pil_image)

# Optional: Display the image to verify the text placement
pil_image.show()