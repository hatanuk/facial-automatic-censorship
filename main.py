import jetson.inference
import jetson.utils
import cv2
import argparse
import time
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str,default="/dev/video0", help="camera feed to process")
parser.add_argument("--output", type=str,default="display://0", help="output URI")
parser.add_argument("--censor-type", type=str,default="blur", help='censor=ship method - can choose between "black-box" and "blur"')
args = parser.parse_args()

net = jetson.inference.detectNet(argv=["--model=model/ssd-mobilenet.onnx", "--lables=model/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes",], threshold=0.35)
display = jetson.utils.videoOutput(args.output) 
camera = jetson.utils.videoSource(args.source)   
info = jetson.utils.cudaFont()


def censor_img(detection, img, type):

  start = time.time()

  top_left = (round(detection.Left), round(detection.Top))
 
  x, y, w, h = int(top_left[0]), int(top_left[1]), int(round(detection.Width)), int(round(detection.Height))


  if type == "black-box":
    #bound ROI with 1s on detection region, 0s elsewhere
    roi = numpy.zeros((img.shape[0], img.shape[1]))
    ones = numpy.ones_like(roi[y : y + h, x : x + w])
    roi[y : y + h, x : x + w] = ones
 
    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        if roi[y, x]: #if corresponds to a 1 in the ROI
          for z in range(img.shape[2]):
            img[y, x, z] = 0 #censor that pixel

  elif type == "blur":
    #convert to cv
    bgr_img = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format="bgr8")
    jetson.utils.cudaConvertColor(img, bgr_img)
    jetson.utils.cudaDeviceSynchronize()
    cv_img = jetson.utils.cudaToNumpy(bgr_img)

    roi = cv_img[y : y + h, x : x + w]

    if roi.any():
  
      blur = cv2.GaussianBlur(roi, (181, 181), 0) # blur roi

      cv_img[y : y + h, x : x + w] = blur #apply blurred region to image

      #convert back to cuda
      bgr_img = jetson.utils.cudaFromNumpy(cv_img, isBGR=True)
      img = jetson.utils.cudaAllocMapped(width=bgr_img.width, height=bgr_img.height, format='rgb8')
      jetson.utils.cudaConvertColor(bgr_img, img)


  end = time.time() - start

  return img, end
  
sum = 0
counter = 0
type = args.censor_type


if args.censor_type != "blur" and args.censor_type != "black-box":
  raise ValueError('censor-type may only be "blur" or "black-box"')

while True:
  img = camera.Capture()
  detections = net.Detect(img, overlay="None")
  

  for detection in detections:
    
    
    img, end = censor_img(detection, img, type)
    if counter < 5:
      counter += 1
      sum += end

    else:
      avg = sum / counter
      print(f"average for {counter} detections: {avg*10**3} ms")
      counter = 0
      sum = 0
 




  display.Render(img)
  display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
