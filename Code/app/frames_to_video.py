import os
import cv2

fps = 25.0
test_videos = [f for f in os.listdir("./Data/Test") if os.path.join("./Data/Test", f)]

for test_video in test_videos:
    pathOut = test_video+'.mp4'
    pathOut = os.path.join("./Data/Test_videos/", pathOut)
    
    pathOut2 = test_video+'_corr'+'.mp4'
    pathOut2 = os.path.join("./Data/Test_videos/", pathOut2)
    
    if pathOut:
        frame_array = []
        for frame_file in sorted(os.listdir(os.path.join("./Data/Test/", test_video))):
            #print(frame_file)
            img = cv2.imread(os.path.join("./Data/Test/", test_video,frame_file))
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        os.system('ffmpeg -i {0} -vcodec libx264 {1}'.format(pathOut,pathOut2))
        os.remove(pathOut)
        