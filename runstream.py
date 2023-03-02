import os
while True:
	os.system("ffmpeg -i raia2_2.mp4 -listen 1 -f mp4 -movflags frag_keyframe+empty_moov http://localhost:8080")
	print("[+] Stream Completed, Now Restarting ...")
