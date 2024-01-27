import cv2
from merged import process_images
from PIL import Image
import numpy as np
import os

# def camm():
# cam = cv2.VideoCapture(2)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# # while cam.isOpened():
# cv2.namedWindow("Camera")
# print("Captured")

# while True:
#     ret, frame = cam.read()

#     if not ret:
#         print("Failed to grab frame")
#         break

#     cv2.imshow("Camera", frame)

#     k = cv2.waitKey(1)

#     # if k % 256 == 27:  # ESC key to exit
#     #     break
#     # elif k % 256 == 32:  # Space key to capture image
#     if k % 256 == 32:  # Space key to capture image
#         # img_name = "./bnet/click/clicked0.png"
#         img_name = "./../BASNet/test/Sabbir.jpg"

#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         cv2.destroyAllWindows()
#         cam.release()

        # Save the original image
        # original_filename = 'original/original_{}.png'.format(
        #     self.img_counter)
        # while os.path.exists(original_filename):
        #     self.img_counter += 1
        #     original_filename = 'original/original_{}.png'.format(
        #         self.img_counter)

        # img = Image.open(img_name)
        # img.save(original_filename, format='png')

       
        # input_dir  = "./bnet/click/"
        # pred_dir = "./bnet/mask/"
        # result_dir = "./bnet/result/"
input_dir  = "./../BASNet/test/"
pred_dir = "./../BASNet/mask/"
result_dir = "./../BASNet/result/"
model_dir = "./saved_models/basnet_bsi/basnet.pth"

process_images(input_dir, pred_dir, result_dir, model_dir)


only_masked_picture_but_transparent = os.path.join(result_dir, f'result0.png')


bg_img = Image.open(self.bg_img_path)
# print("bg_img size before resize:", bg_img.size)

foreground_img = Image.open(only_masked_picture_but_transparent)
bg_img = bg_img.resize((foreground_img.width, foreground_img.height))
# new_bg_path.show()
result = cv2.imread(foreground_img)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
        # bg_bg = "./bg7.png"
        

        # new_bg = Image.open(new_bg_path).convert("RGBA")
        # bg_bg = Image.open(bg_bg).convert("RGBA")
        # new_bg = new_bg.resize((bg_bg.width, bg_bg.height))

        # # Calculate the position to paste the new_bg image at the center
        # center_x = (bg_bg.width - new_bg.width) // 2
        # center_y = (bg_bg.height - new_bg.height) // 2
        # position = (center_x, center_y)

        # # Composite the images
        # result = Image.alpha_composite(bg_bg, Image.new("RGBA", bg_bg.size, (0, 0, 0, 0)))
        # result.paste(new_bg, position, new_bg)

        # result_path = './bnet/result/composite_result.png'
        # result.save(result_path)

        # # Save or display the result
        # result.show()
        # frame = "./ff2.png"
        # result_path = './bnet/result/composite_result.png'

        # open_frame = Image.open(frame).convert("RGBA")
        # open_oldBG = Image.open(result_path).convert("RGBA")

        # center_x = (open_oldBG.width - open_frame.width) // 2
        # center_y = (open_oldBG.height - open_frame.height) // 2
        # position = (center_x, center_y)

        # # Paste the frame onto the open_oldBG at the calculated position
        # open_oldBG.paste(open_frame, position, open_frame)

        # # Save the result
        # final_result_path = './bnet/result/final_result.png'
        # open_oldBG.save(final_result_path)

        # # Display or further process the result as needed
        # open_oldBG.show()





# result = Image.alpha_composite(bg_bg, Image.new("RGBA", bg_bg.size, (0, 0, 0, 0)))
# result.paste(new_bg, position, new_bg)


# result = cv2.imread(result_path)
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


        # Remove the background
     

        # Load the background image
        # bg_path = './bg7.png'
        # bg_img = Image.open(bg_path).convert("RGBA")
        # bg_img = bg_img.resize((bg_path.width, bg_path.height))

        # # # Paste foreground onto the BG
        # bg_img = Image.open(bg_path)
        # bg_img = bg_img.resize((img.width, img.height))

        # # result_img.save(frame_path, format='PNG')
        # bg_img.paste(masked_filename, (0, 0), masked_filename)

        # frame_path = './ff2.png'
        # # Load the frame image
        # frame_img = Image.open()
        # frame_img.show()
        # Ensure the foreground image has an alpha channel
        # if 'A' not in frame_img.getbands():
        #     frame_img.putalpha(255)  # Add a fully opaque alpha channel

        # # Calculate the position to paste the foreground image at the center
        # position = ((bg_img.width - frame_img.width) // 2, (bg_img.height - frame_img.height) // 2)

        # bg_img.paste(frame_img, (0, 0), frame_img)