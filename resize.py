#!/usr/bin/env python
# coding: utf-8

# In[52]:


from PIL import Image
import os.path
import glob

def convertjpg(jpgfile,outdir):
    img=Image.open(jpgfile)
    img.thumbnail((1008, 1008))  # 目前我先把圖片縮小成原先的1/4 ,會變成1008 * 756 的大小
    img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    print(img)

for jpgfile in glob.glob(r"D:\\ytl\\CYCU\\專題\\weed\\kaggle\\*.jpg"):
    convertjpg(jpgfile, r"D:\\ytl\\CYCU\\專題\\weed\\kaggle\\resize")

