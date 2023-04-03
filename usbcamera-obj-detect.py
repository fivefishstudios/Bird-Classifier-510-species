# usbcamera-obj-detect.py
# 4/1/23
# Bird Feeder Camera with AI Detection and Recording

import cv2 
import numpy as np 
import torch
import seaborn as sn
import os
import datetime 
import time
import requests
import zipfile
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchinfo import summary
from PIL import Image
from joblib.externals.loky.backend.context import get_context

# setup cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# object detection model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)

# model location for ResNet101 
MODEL_PATH = Path("models")
MODEL_NAME = "simple-bird-resnet101-510species.pth"
MODEL_PATH_NAME = MODEL_PATH / MODEL_NAME

# NOTE: 510 Bird Species
class_names = ['ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL', 'AFRICAN CROWNED CRANE', 
               'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH', 'AFRICAN OYSTER CATCHER', 'AFRICAN PIED HORNBILL', 
               'AFRICAN PYGMY GOOSE', 'ALBATROSS', 'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH', 
               'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN FLAMINGO', 
               'AMERICAN GOLDFINCH', 'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART', 'AMERICAN ROBIN', 
               'AMERICAN WIGEON', 'AMETHYST WOODSTAR', 'ANDEAN GOOSE', 'ANDEAN LAPWING', 'ANDEAN SISKIN', 
               'ANHINGA', 'ANIANIAU', 'ANNAS HUMMINGBIRD', 'ANTBIRD', 'ANTILLEAN EUPHONIA', 'APAPANE', 
               'APOSTLEBIRD', 'ARARIPE MANAKIN', 'ASHY STORM PETREL', 'ASHY THRUSHBIRD', 'ASIAN CRESTED IBIS', 
               'ASIAN DOLLARD BIRD', 'ASIAN GREEN BEE EATER', 'AUCKLAND SHAQ', 'AUSTRAL CANASTERO', 'AUSTRALASIAN FIGBIRD', 
               'AVADAVAT', 'AZARAS SPINETAIL', 'AZURE BREASTED PITTA', 'AZURE JAY', 'AZURE TANAGER', 'AZURE TIT', 
               'BAIKAL TEAL', 'BALD EAGLE', 'BALD IBIS', 'BALI STARLING', 'BALTIMORE ORIOLE', 'BANANAQUIT', 'BAND TAILED GUAN', 'BANDED BROADBILL', 'BANDED PITA', 'BANDED STILT', 'BAR-TAILED GODWIT', 'BARN OWL', 'BARN SWALLOW', 'BARRED PUFFBIRD', 'BARROWS GOLDENEYE', 'BAY-BREASTED WARBLER', 'BEARDED BARBET', 'BEARDED BELLBIRD', 'BEARDED REEDLING', 'BELTED KINGFISHER', 'BIRD OF PARADISE', 'BLACK AND YELLOW BROADBILL', 'BLACK BAZA', 'BLACK COCKATO', 'BLACK FACED SPOONBILL', 'BLACK FRANCOLIN', 'BLACK HEADED CAIQUE', 'BLACK NECKED STILT', 'BLACK SKIMMER', 'BLACK SWAN', 'BLACK TAIL CRAKE', 'BLACK THROATED BUSHTIT', 'BLACK THROATED HUET', 'BLACK THROATED WARBLER', 'BLACK VENTED SHEARWATER', 'BLACK VULTURE', 'BLACK-CAPPED CHICKADEE', 'BLACK-NECKED GREBE', 'BLACK-THROATED SPARROW', 'BLACKBURNIAM WARBLER', 'BLONDE CRESTED WOODPECKER', 'BLOOD PHEASANT', 'BLUE COAU', 'BLUE DACNIS', 'BLUE GRAY GNATCATCHER', 'BLUE GROSBEAK', 'BLUE GROUSE', 'BLUE HERON', 'BLUE MALKOHA', 'BLUE THROATED TOUCANET', 'BOBOLINK', 'BORNEAN BRISTLEHEAD', 'BORNEAN LEAFBIRD', 'BORNEAN PHEASANT', 'BRANDT CORMARANT', 'BREWERS BLACKBIRD', 'BROWN CREPPER', 'BROWN HEADED COWBIRD', 'BROWN NOODY', 'BROWN THRASHER', 'BUFFLEHEAD', 'BULWERS PHEASANT', 'BURCHELLS COURSER', 'BUSH TURKEY', 'CAATINGA CACHOLOTE', 'CACTUS WREN', 'CALIFORNIA CONDOR', 'CALIFORNIA GULL', 'CALIFORNIA QUAIL', 'CAMPO FLICKER', 'CANARY', 'CANVASBACK', 'CAPE GLOSSY STARLING', 'CAPE LONGCLAW', 'CAPE MAY WARBLER', 'CAPE ROCK THRUSH', 'CAPPED HERON', 'CAPUCHINBIRD', 'CARMINE BEE-EATER', 'CASPIAN TERN', 'CASSOWARY', 'CEDAR WAXWING', 'CERULEAN WARBLER', 'CHARA DE COLLAR', 'CHATTERING LORY', 'CHESTNET BELLIED EUPHONIA', 'CHINESE BAMBOO PARTRIDGE', 'CHINESE POND HERON', 'CHIPPING SPARROW', 'CHUCAO TAPACULO', 'CHUKAR PARTRIDGE', 'CINNAMON ATTILA', 'CINNAMON FLYCATCHER', 'CINNAMON TEAL', 'CLARKS GREBE', 'CLARKS NUTCRACKER', 'COCK OF THE  ROCK', 'COCKATOO', 'COLLARED ARACARI', 'COLLARED CRESCENTCHEST', 'COMMON FIRECREST', 'COMMON GRACKLE', 'COMMON HOUSE MARTIN', 'COMMON IORA', 'COMMON LOON', 'COMMON POORWILL', 'COMMON STARLING', 'COPPERSMITH BARBET', 'COPPERY TAILED COUCAL', 'CRAB PLOVER', 'CRANE HAWK', 'CREAM COLORED WOODPECKER', 'CRESTED AUKLET', 'CRESTED CARACARA', 'CRESTED COUA', 'CRESTED FIREBACK', 'CRESTED KINGFISHER', 'CRESTED NUTHATCH', 'CRESTED OROPENDOLA', 'CRESTED SERPENT EAGLE', 'CRESTED SHRIKETIT', 'CRESTED WOOD PARTRIDGE', 'CRIMSON CHAT', 'CRIMSON SUNBIRD', 'CROW', 'CROWNED PIGEON', 'CUBAN TODY', 'CUBAN TROGON', 'CURL CRESTED ARACURI', 'D-ARNAUDS BARBET', 'DALMATIAN PELICAN', 'DARJEELING WOODPECKER', 'DARK EYED JUNCO', 'DAURIAN REDSTART', 'DEMOISELLE CRANE', 'DOUBLE BARRED FINCH', 'DOUBLE BRESTED CORMARANT', 'DOUBLE EYED FIG PARROT', 'DOWNY WOODPECKER', 'DUSKY LORY', 'DUSKY ROBIN', 'EARED PITA', 'EASTERN BLUEBIRD', 'EASTERN BLUEBONNET', 'EASTERN GOLDEN WEAVER', 'EASTERN MEADOWLARK', 'EASTERN ROSELLA', 'EASTERN TOWEE', 'EASTERN WIP POOR WILL', 'EASTERN YELLOW ROBIN', 'ECUADORIAN HILLSTAR', 'EGYPTIAN GOOSE', 'ELEGANT TROGON', 'ELLIOTS  PHEASANT', 'EMERALD TANAGER', 'EMPEROR PENGUIN', 'EMU', 'ENGGANO MYNA', 'EURASIAN BULLFINCH', 'EURASIAN GOLDEN ORIOLE', 'EURASIAN MAGPIE', 'EUROPEAN GOLDFINCH', 'EUROPEAN TURTLE DOVE', 'EVENING GROSBEAK', 'FAIRY BLUEBIRD', 'FAIRY PENGUIN', 'FAIRY TERN', 'FAN TAILED WIDOW', 'FASCIATED WREN', 'FIERY MINIVET', 'FIORDLAND PENGUIN', 'FIRE TAILLED MYZORNIS', 'FLAME BOWERBIRD', 'FLAME TANAGER', 'FOREST WAGTAIL', 'FRIGATE', 'FRILL BACK PIGEON', 'GAMBELS QUAIL', 'GANG GANG COCKATOO', 'GILA WOODPECKER', 'GILDED FLICKER', 'GLOSSY IBIS', 'GO AWAY BIRD', 'GOLD WING WARBLER', 'GOLDEN BOWER BIRD', 'GOLDEN CHEEKED WARBLER', 'GOLDEN CHLOROPHONIA', 'GOLDEN EAGLE', 'GOLDEN PARAKEET', 'GOLDEN PHEASANT', 'GOLDEN PIPIT', 'GOULDIAN FINCH', 'GRANDALA', 'GRAY CATBIRD', 'GRAY KINGBIRD', 'GRAY PARTRIDGE', 'GREAT ARGUS', 'GREAT GRAY OWL', 'GREAT JACAMAR', 'GREAT KISKADEE', 'GREAT POTOO', 'GREAT TINAMOU', 'GREAT XENOPS', 'GREATER PEWEE', 'GREATER PRAIRIE CHICKEN', 'GREATOR SAGE GROUSE', 'GREEN BROADBILL', 'GREEN JAY', 'GREEN MAGPIE', 'GREEN WINGED DOVE', 'GREY CUCKOOSHRIKE', 'GREY HEADED FISH EAGLE', 'GREY PLOVER', 'GROVED BILLED ANI', 'GUINEA TURACO', 'GUINEAFOWL', 'GURNEYS PITTA', 'GYRFALCON', 'HAMERKOP', 'HARLEQUIN DUCK', 'HARLEQUIN QUAIL', 'HARPY EAGLE', 'HAWAIIAN GOOSE', 'HAWFINCH', 'HELMET VANGA', 'HEPATIC TANAGER', 'HIMALAYAN BLUETAIL', 'HIMALAYAN MONAL', 'HOATZIN', 'HOODED MERGANSER', 'HOOPOES', 'HORNED GUAN', 'HORNED LARK', 'HORNED SUNGEM', 'HOUSE FINCH', 'HOUSE SPARROW', 'HYACINTH MACAW', 'IBERIAN MAGPIE', 'IBISBILL', 'IMPERIAL SHAQ', 'INCA TERN', 'INDIAN BUSTARD', 'INDIAN PITTA', 'INDIAN ROLLER', 'INDIAN VULTURE', 'INDIGO BUNTING', 'INDIGO FLYCATCHER', 'INLAND DOTTEREL', 'IVORY BILLED ARACARI', 'IVORY GULL', 'IWI', 'JABIRU', 'JACK SNIPE', 'JACOBIN PIGEON', 'JANDAYA PARAKEET', 'JAPANESE ROBIN', 'JAVA SPARROW', 'JOCOTOCO ANTPITTA', 'KAGU', 'KAKAPO', 'KILLDEAR', 'KING EIDER', 'KING VULTURE', 'KIWI', 'KOOKABURRA', 'LARK BUNTING', 'LAUGHING GULL', 'LAZULI BUNTING', 'LESSER ADJUTANT', 'LILAC ROLLER', 'LIMPKIN', 'LITTLE AUK', 'LOGGERHEAD SHRIKE', 'LONG-EARED OWL', 'LOONEY BIRDS', 'LUCIFER HUMMINGBIRD', 'MAGPIE GOOSE', 'MALABAR HORNBILL', 'MALACHITE KINGFISHER', 'MALAGASY WHITE EYE', 'MALEO', 'MALLARD DUCK', 'MANDRIN DUCK', 'MANGROVE CUCKOO', 'MARABOU STORK', 'MASKED BOBWHITE', 'MASKED BOOBY', 'MASKED LAPWING', 'MCKAYS BUNTING', 'MERLIN', 'MIKADO  PHEASANT', 'MILITARY MACAW', 'MOURNING DOVE', 'MYNA', 'NICOBAR PIGEON', 'NOISY FRIARBIRD', 'NORTHERN BEARDLESS TYRANNULET', 'NORTHERN CARDINAL', 'NORTHERN FLICKER', 'NORTHERN FULMAR', 'NORTHERN GANNET', 'NORTHERN GOSHAWK', 'NORTHERN JACANA', 'NORTHERN MOCKINGBIRD', 'NORTHERN PARULA', 'NORTHERN RED BISHOP', 'NORTHERN SHOVELER', 'OCELLATED TURKEY', 'OKINAWA RAIL', 'ORANGE BRESTED BUNTING', 'ORIENTAL BAY OWL', 'ORNATE HAWK EAGLE', 'OSPREY', 'OSTRICH', 'OVENBIRD', 'OYSTER CATCHER', 'PAINTED BUNTING', 'PALILA', 'PALM NUT VULTURE', 'PARADISE TANAGER', 'PARAKETT  AKULET', 'PARUS MAJOR', 'PATAGONIAN SIERRA FINCH', 'PEACOCK', 'PEREGRINE FALCON', 'PHAINOPEPLA', 'PHILIPPINE EAGLE', 'PINK ROBIN', 'PLUSH CRESTED JAY', 'POMARINE JAEGER', 'PUFFIN', 'PUNA TEAL', 'PURPLE FINCH', 'PURPLE GALLINULE', 'PURPLE MARTIN', 'PURPLE SWAMPHEN', 'PYGMY KINGFISHER', 'PYRRHULOXIA', 'QUETZAL', 'RAINBOW LORIKEET', 'RAZORBILL', 'RED BEARDED BEE EATER', 'RED BELLIED PITTA', 'RED BILLED TROPICBIRD', 'RED BROWED FINCH', 'RED FACED CORMORANT', 'RED FACED WARBLER', 'RED FODY', 'RED HEADED DUCK', 'RED HEADED WOODPECKER', 'RED KNOT', 'RED LEGGED HONEYCREEPER', 'RED NAPED TROGON', 'RED SHOULDERED HAWK', 'RED TAILED HAWK', 'RED TAILED THRUSH', 'RED WINGED BLACKBIRD', 'RED WISKERED BULBUL', 'REGENT BOWERBIRD', 'RING-NECKED PHEASANT', 'ROADRUNNER', 'ROCK DOVE', 'ROSE BREASTED COCKATOO', 'ROSE BREASTED GROSBEAK', 'ROSEATE SPOONBILL', 'ROSY FACED LOVEBIRD', 
               # ....
               'ROUGH LEG BUZZARD', 'ROYAL FLYCATCHER', 'RUBY CROWNED KINGLET', 'RUBY THROATED HUMMINGBIRD', 
               'RUDDY SHELDUCK', 'RUDY KINGFISHER', 'RUFOUS KINGFISHER', 'RUFOUS TREPE', 'RUFUOS MOTMOT', 
               'SAMATRAN THRUSH', 'SAND MARTIN', 'SANDHILL CRANE', 'SATYR TRAGOPAN', 'SAYS PHOEBE', 
               'SCARLET CROWNED FRUIT DOVE', 'SCARLET FACED LIOCICHLA', 'SCARLET IBIS', 'SCARLET MACAW', 
               'SCARLET TANAGER', 'SHOEBILL', 'SHORT BILLED DOWITCHER', 'SMITHS LONGSPUR', 'SNOW GOOSE', 
               'SNOWY EGRET', 'SNOWY OWL', 'SNOWY PLOVER', 'SORA', 'SPANGLED COTINGA', 'SPLENDID WREN', 
               'SPOON BILED SANDPIPER', 'SPOTTED CATBIRD', 'SPOTTED WHISTLING DUCK', 'SQUACCO HERON', 
               'SRI LANKA BLUE MAGPIE', 'STEAMER DUCK', 'STORK BILLED KINGFISHER', 'STRIATED CARACARA', 
               'STRIPED OWL', 'STRIPPED MANAKIN', 'STRIPPED SWALLOW', 'SUNBITTERN', 'SUPERB STARLING', 'SURF SCOTER', 'SWINHOES PHEASANT', 'TAILORBIRD', 'TAIWAN MAGPIE', 'TAKAHE', 'TASMANIAN HEN', 'TAWNY FROGMOUTH', 'TEAL DUCK', 'TIT MOUSE', 'TOUCHAN', 'TOWNSENDS WARBLER', 'TREE SWALLOW', 'TRICOLORED BLACKBIRD', 'TROPICAL KINGBIRD', 'TRUMPTER SWAN', 'TURKEY VULTURE', 'TURQU-14.4184OISE MOTMOT', 'UMBRELLA BIRD', 'VARIED THRUSH', 'VEERY', 'VENEZUELIAN TROUPIAL', 'VERDIN', 'VERMILION FLYCATHER', 'VICTORIA CROWNED PIGEON', 'VIOLET BACKED STARLING', 'VIOLET GREEN SWALLOW', 'VIOLET TURACO', 'VISAYAN HORNBILL', 'VULTURINE GUINEAFOWL', 'WALL CREAPER', 'WATTLED CURASSOW', 'WATTLED LAPWING', 'WHIMBREL', 'WHITE BREASTED WATERHEN', 'WHITE BROWED CRAKE', 'WHITE CHEEKED TURACO', 'WHITE CRESTED HORNBILL', 'WHITE EARED HUMMINGBIRD', 'WHITE NECKED RAVEN', 'WHITE TAILED TROPIC', 'WHITE THROATED BEE EATER', 'WILD TURKEY', 'WILLOW PTARMIGAN', 
               # ...
               'WILSONS BIRD OF PARADISE', 'WOOD DUCK', 'WOOD THRUSH', 'WOODLAND KINGFISHER', 'WRENTIT', 
               'YELLOW BELLIED FLOWERPECKER', 'YELLOW CACIQUE', 'YELLOW HEADED BLACKBIRD', 'ZEBRA DOVE']

isCapturingVideo = False 
birdFound = False 
birdIsIdentified = False  
birdSpeciesName = ''

MaximumVideoLength = 6  # in seconds
videoOutputPath = "./videos/"

# data_transform = transforms.Compose([  # Compose is used to serialize our tranforms functions
#     # transform operations are a list
#     # 1. resize image to be smaller
#     transforms.Resize(size=(224, 224), antialias=True),
#     # 2. do some data augmentation
#     transforms.RandomHorizontalFlip(p=0.25),  # flip horizontal 25% of the time
#     transforms.RandomRotation(degrees=30),  # test_acc = 0.6710
#     # transforms.RandomAffine(degrees=15),      # test_acc = 0.6347
#     # 3. convert to Tensor so PyTorch can use the data
#     transforms.ToTensor(),  # ultimate goal is to convert to Tensors
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop(720),
    transforms.CenterCrop(448),
    transforms.Resize(size=(224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# load our bird classifier
birdClassifierModel  = torchvision.models.resnet101().to(device)
# override classifier method in model class
# Recreate the classifier layer and seed it to the target device
birdClassifierModel.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,  # same number of output units as our number of classes
                    )).to(device)
                    # bias=True)).to(device)
# load model from disk
birdClassifierModel.load_state_dict(torch.load(f=MODEL_PATH_NAME))            
    


# open the file (or the video usb camera)
usbCameraDevice = 0   # first USB camera device
usbCamera = cv2.VideoCapture(usbCameraDevice)

# set new size of video capture (WxH combination doesn't always work)
ret = usbCamera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
ret = usbCamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# read and display one frame
ret, frame = usbCamera.read()

# if we're resaving this video, create a VideoWriter 
videoformat = cv2.VideoWriter_fourcc(*'mp4v')     # .mp4
fps = 15  # 15fps seems to be the normal speed 
width  = int(usbCamera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(usbCamera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
framesize = (width, height)

def CloseVideoFile():
    # close currently opened video file 
    videoOutput.release()
    # destroy last window
    cv2.destroyAllWindows()

# while file (or camera) is open, do the following loop
while usbCamera.isOpened():
    # print("reading new frame from usb camera...")
    # read frame of video file
    ret, frame = usbCamera.read()
    
    # FOR DEBUGGING
    # filepathname = "./unknown/phil-eagle1.jpg"
    # filepathname = "./unknown/mourning-dove1.jpg"
    # frame = cv2.imread(filepathname)
     
    # is frame valid? check the return value
    if not ret:
        print("USB Camera not found... exiting")
        break 

    # DEBUG: preview video frame in window 
    # cv2.imshow("DEBUG", frame)
    
    # adjust brightness, i.e. normalize 
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        
    if not isCapturingVideo:   
        # do object detection of usb camera frame capture
        results = model(frame)
        # check if bird object detected 
        df = results.pandas().xyxy[0] 
        if not df.empty: # logic guard if no object detected
            objdetected = df['name'].values[0]
            # check if bird detected 
            if objdetected == 'bird':
                print(f'object detected: {objdetected}')
                BirdFound = True 
                # get new frame from camera again 
                # time.sleep(0.30)
                # ret, frame = usbCamera.read()
                # let's try to identify the bird species
                # openCV uses BGR so we need to convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # resize image to 224x224, same size expected by our model
                frame_t = img_transform(frame_rgb)  # convert to tensor
                frame_t = frame_t.unsqueeze(dim=0)  # add batch size
                frame_t = frame_t.to(device)
                birdClassifierModel.eval()
                with torch.inference_mode():
                    predicted_class_logits = birdClassifierModel(frame_t)
                    # print(f"\nlogits: {predicted_class_logits}")
                    birdSpeciesName = class_names[torch.argmax(predicted_class_logits)]           
                    birdIsIdentified = True 
                    print(f'bird identified as {birdSpeciesName}')
                
                # build filepath name so we can save this new video   
                datetimenow = (datetime.datetime.now()).strftime("%c")
                filenameOut = videoOutputPath + birdSpeciesName + datetimenow + ".mp4"  
                print(f"creating video file: {filenameOut}")
                videoOutput = cv2.VideoWriter(filenameOut, videoformat, fps, framesize)

                # set flag that we want to start capturing video
                isCapturingVideo = True 
                
                # we'll record MaximumVideoLength second clips for each bird capture
                start_time = timer()    
        else:
            # print('object recognition yolo failed!')
            # print(df)
            cv2.destroyAllWindows()
            pass  # do nothing 
        
    # check if we're starting or in middle of recording video
    if isCapturingVideo:
        # check if bird flew away
        results = model(frame)
        df = results.pandas().xyxy[0] 
        if not df.empty: # logic guard if no object detected
            objdetected = df['name'].values[0]
            # check if bird detected 
            if objdetected == 'bird':
                pass  # continue 
            else:
                # bird flew away 
                isCapturingVideo = False 
                birdIsIdentified = False 
                birdSpeciesName = '' 
                birdFound = False 
                CloseVideoFile()
                cv2.destroyAllWindows()
                continue 
                
        # write birdspecies text on video frame
        # text shadow
        cv2.putText(frame, birdSpeciesName, (33, 53), cv2.FONT_HERSHEY_SIMPLEX, 1.12, (20, 20, 20), 3)
        # text foreground color
        cv2.putText(frame, birdSpeciesName, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.12, (20, 255, 20), 3)
        
        # preview video frame in window 
        cv2.imshow(f"{birdSpeciesName}", frame)
        
        # write the frame to .mp4 file 
        videoOutput.write(frame)
        
        # check if MaxX seconds had passed
        if (timer() - start_time > MaximumVideoLength):
            print(f"{MaximumVideoLength} seconds elapsed, closing video file {filenameOut}\n")
            # stop recording, close this video file, get ready for next capture
            isCapturingVideo = False 
            birdIsIdentified = False 
            birdSpeciesName = '' 
            birdFound = False 
            CloseVideoFile()
            cv2.destroyAllWindows()
            continue # restart loop 
        
        
            
    # check for keypress 'q' to exit
    if cv2.waitKey(10) == ord('q') :
        print("closing USB Camera... end of program!")
        # close camera file 
        usbCamera.release() 
        # close any window created
        cv2.destroyAllWindows()
        break # exit loop 

