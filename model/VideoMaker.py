def Resize( fileName ) :
	import moviepy.editor as mp
	clip = mp.VideoFileClip(fileName + '.mp4')
	clip_resized = clip.resize((640, 480)) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
	clip_resized.write_videofile(fileName+'_resize.mp4')

def ChangeFPS( fileName ) :
	import cv2

	outfile_name = fileName + '_30FPS.mp4'

	images = []
	frame_now = 1
	take = True
	cap = cv2.VideoCapture( fileName+'.mp4' )
	while cap.isOpened() :
		ret, frame = cap.read() # ret is whether read is succesful, frame is return frame
		if ret :
			if take :
				images.append( frame )
				take = False
			else :
				take = True
				
			print( frame_now )
			frame_now += 1
			
		else :
			break
			
	height,width,layers=images[0].shape
	out_video=cv2.VideoWriter(outfile_name,-1,25, frameSize=(width, height) )
	for i in images :
		out_video.write( i )
		
	out_video.release()
	cap.release()
	print( 'Done' )

def imgToMp4 :
	import cv2
	import os

	path = '.\\'
	allFileList = os.listdir()

	for num in allFileList :
		path = '.\\'
		if os.path.isdir(os.path.join(path,num)):
			path = path + num + '\\' + 'rgb'
			print("I'm a directory: ", path )
		
			imgs = []
			imgs_path = os.listdir( path )
			for img_path in imgs_path :
				img = cv2.imread( path + '\\' + img_path )
				imgs.append( img ) 
			
			#height,width,layers=imgs[1].shape
			video=cv2.VideoWriter( '.\\' + num + '.mp4',-1,25, frameSize=(320, 240) )
			for img in imgs:
				video.write(img)
				
			video.release()
			print( 'done:', num )
		
	
fileName = input( 'Enter a file name\n> ' )
Resize( fileName )