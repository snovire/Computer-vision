import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt
import json,pickle
from tqdm import tqdm_notebook,tqdm
import sys
import urllib.request as request
import pickle
import copy
import json

# Gloabal Functions
def integral_image(img):
	row=img.shape[0]
	col=img.shape[1]
	img_integ=np.int64(np.copy(img))
	for i in range(0,row):
		for j in range(0,col):
			if i==0 and j>0:
				L=img_integ[i,j-1] # integral value on left pixel
				U=0 # integral value on up pixel
				D=0 # integral value on left-up diagonal pixel
			elif i>0 and j==0:
				L=0
				U=img_integ[i-1,j]
				D=0
			elif i==0 and j==0:
				L=0
				U=0
				D=0
			else:
				L=img_integ[i,j-1]
				U=img_integ[i-1,j]
				D=img_integ[i-1,j-1]
			img_integ[i,j]=L+U-D+img_integ[i,j]
	return img_integ
def cal_integral_image_dict(images):
	if isinstance(images,dict):
		inte_dict={}
		for key,val in tqdm(images.items()):
			inte=integral_image(val)
			inte_dict[key]=inte
	else:
		inte_dict=np.zeros(images.shape)
		for i in tqdm(range(len(images))):
			inte_dict[i,:,:]=integral_image(images[i,:,:])
	return inte_dict
def read_target_region(path): # Exatract face region from images
	files=os.listdir(path)
	files=sorted(files)
	feat_dict={}
	for file in files:
		name=file.split('.')
		if name[1]=="txt" and name[0][len(name[0])-4:]!="List": # not ellipseList file
			with open(path+'/'+file) as f:
				for line in f:
					line=line.strip()
					if line not in feat_dict:
						feat_dict[line]=[]
					else:
						print("Already exist key: ",line)
			f.close()
	for file in files:
		name=file.split('.')
		if name[1]=="txt" and name[0][len(name[0])-4:]=="List": # ellipseList file
			with open(path+'/'+file) as f:
				num_obj=-1
				for line in f:
					line=line.strip()
					if line in feat_dict:
						read_num=True
						read_obj=False
						key=line
					elif read_num:
						num_obj=int(line)
						feat_dict[key]=np.zeros((num_obj,6))
						obj_index=0
						read_num=False
						read_obj=True
					elif read_obj:
						line=[float(l) for l in line.split()]
						feat_dict[key][obj_index]=line
						obj_index+=1
						if obj_index>=num_obj:
							read_obj=False          
			f.close()
	return feat_dict
def read_image(imgidx, path, style="Gray"):
	img_dict={}
	for idx in imgidx:
		fullpath=path+idx+".jpg"
		if style=="Gray":
			img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
		elif style=="RGB":
			img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
		if not img.dtype == np.uint8:
			pass
		img_dict[idx]=img
	return img_dict
def judge_inregion(region, rect_q, regiontype="ellipse"):
	"""
	rect_q: an array of query rectangles, each row is [left-up x, left-up y, right-bottom x, right-bottom y]
	region: a ground truth region where a face is located
	"""
	if regiontype=="ellipse":
		dy,dx,angle,x_c_tr,y_c_tr,score=region
		x_lu_tr=x_c_tr-dx # scalar
		y_lu_tr=y_c_tr-dy
		x_rb_tr=x_c_tr+dx
		y_rb_tr=y_c_tr+dy    
	x_lu_q,y_lu_q,x_rb_q,y_rb_q=rect_q.T # Left-up right-bottom coordinates of queried features.Mulit-Dim array
	# Feature is positive if feature rectangle is in target region,and rectangle covers center of target region
	LUinregion=(x_lu_q>=x_lu_tr) & (x_lu_q<x_rb_tr) & (y_lu_q>=y_lu_tr) & (y_lu_q<y_rb_tr)
	RBinregion=(x_rb_q>=x_lu_tr) & (x_rb_q<x_rb_tr) & (y_rb_q>=y_lu_tr) & (y_rb_q<y_rb_tr)
	centerinquery=(x_c_tr>=x_lu_q)&(x_c_tr<x_rb_q)&(y_c_tr>=y_lu_q)&(y_c_tr<y_rb_q)
	return LUinregion & RBinregion & centerinquery
def draw_rectangle_v2(img, imgmark): # draw image with rectangles. Multitype inputs
	img_out=np.copy(img)
	rmax=img.shape[0]-1
	cmax=img.shape[1]-1
	if isinstance(imgmark,list):
		for l in imgmark:
			for mark in l:
				u=min(max(0, int(mark[0])),cmax)
				v=min(max(0, int(mark[1])),rmax)
				ud=min(max(0, int(mark[2])),cmax)
				vd=min(max(0, int(mark[3])),rmax)
				img_out[v:vd,[u,ud]]=255
				img_out[[v,vd],u:ud]=255
	else: # numpy array
		for mark in imgmark:
			u=min(max(0, int(mark[0])),cmax)
			v=min(max(0, int(mark[1])),rmax)
			ud=min(max(0, int(mark[2])),cmax)
			vd=min(max(0, int(mark[3])),rmax)

			img_out[v:vd,[u,ud]]=255
			img_out[[v,vd],u:ud]=255
	return img_out
def draw_rectangle(img, imgmark):
	img_out=np.copy(img)
	rmax=img.shape[0]-1
	cmax=img.shape[1]-1
	for mark in imgmark:
		u=int(mark[3])
		v=int(mark[4])
		dv=int(mark[0]) # first is the shift of row number from its centor
		du=int(mark[1]) # second is the shift of column number from its centor
		img_out[max(v-dv,0):min(v+dv+1,rmax),[max(u-du,0),min(u+du,cmax)]]=255
		img_out[[max(v-dv,0),min(v+dv,rmax)],max(u-du,0):min(u+du+1,cmax)]=255
	return img_out
def split_image(images, regions, win_noface):
	"""
	win_noface: window size=[column, row]
	"""
	dr=win_noface[1]
	dc=win_noface[0]
	dic_face={}
	dic_noface={}
	num_face=0
	num_noface=0
	for key in regions.keys(): # image id as key
		facearr=regions[key]
		rmax=images[key].shape[0]-1
		cmax=images[key].shape[1]-1
		face=np.zeros((len(facearr),4),dtype=np.int)
		for i in range(facearr.shape[0]):
			dy,dx,angle,x_c,y_c,score=facearr[i]
			x_lu=max(x_c-dx,0)# scalar
			y_lu=max(y_c-dy,0)
			x_rb=min(x_c+dx,cmax)
			y_rb=min(y_c+dy,rmax) 
			face[i,:]=[int(x_lu),int(y_lu),int(x_rb),int(y_rb)]
			num_face+=1
		dic_face[key]=face
		# Split non-face patch
		noface=[]
		nr=int((rmax+1)/dr)
		nc=int((cmax+1)/dc)
		for p in range(0,nr):
			y_lu_n=p*dr
			y_rb_n=y_lu_n+dr-1
			for q in range(0,nc):
				x_lu_n=q*dc
				x_rb_n=x_lu_n+dc-1
#                 logbit1=(y_lu_n>=face[:,1])&(y_lu_n<=face[:,3])& (((x_lu_n>=face[:,0])&(x_lu_n<=face[:,2]))|((x_rb_n>=face[:,0])&(x_rb_n<=face[:,2])))
#                 logbit2=(x_lu_n>=face[:,0])&(x_lu_n<=face[:,2])& (((y_lu_n>=face[:,1])&(y_lu_n<=face[:,3]))|((y_rb_n>=face[:,1])&(y_rb_n<=face[:,3])))
#                 logbit=logbit1|logbit2
				logbit=(face[:,0]<=x_rb_n)&(x_lu_n<=face[:,2])&(face[:,1]<=y_rb_n)&(y_lu_n<=face[:,3])
				if not np.any(logbit): # any of logbit is true,non-face patch overlaps face patch in this image
					noface.append([x_lu_n,y_lu_n,x_rb_n,y_rb_n])
					num_noface+=1
		dic_noface[key]=np.array(noface,dtype=np.int)
	return dic_face, dic_noface, num_face, num_noface
def cal_face_boundary(regions):
	idximg=list(regions.keys())
	cmax=-1
	rmax=-1
	cmin=1e15
	rmin=1e15
	idextrem=["","","",""]
	for i in range(len(idximg)):
		face=regions[idximg[i]]
		for j in range(len(face)):
			dy,dx,angle,x_c,y_c,score=face[j,:]
			c=2*dx+1
			r=2*dy+1
			if cmax<c:
				cmax=c
				idextrem[3]=idximg[i]
			if rmax<r:
				rmax=r
				idextrem[1]=idximg[i]
			if cmin>c and c>4:
				cmin=c
				idextrem[2]=idximg[i]
			if rmin>r and r>4:
				rmin=r
				idextrem[0]=idximg[i]
	return (rmin,rmax, cmin,cmax),idextrem
def get_train_data(images, regions, win_noface, scalesize):
	"""
	win_noface=(column number, row number)
	scalesize=(column number, row number)
	win_noface should be larger or equal to scalesize, scalesize is the final images size used in feature extraction
	"""
	dic_face,dic_noface,num_face,num_noface=split_image(images, regions, win_noface)# computer coordiantes of face and non face
	imgids=list(regions.keys())
	face=np.zeros((num_face, scalesize[1],scalesize[0]))
	noface=np.zeros((num_noface, scalesize[1],scalesize[0]))
	ctr_face=0
	ctr_noface=0
	for imgid in imgids:
		coor_face=dic_face[imgid] # rectangle coordinates of face in each row
		coor_noface=dic_noface[imgid] # rectangle of non-face in each row
		img=images[imgid]
		for i in range(len(coor_face)):
			temp=img[coor_face[i,1]:coor_face[i,3],coor_face[i,0]:coor_face[i,2]]
			face[ctr_face,:,:]=cv2.resize(temp, scalesize)
			ctr_face+=1
		for j in range(len(coor_noface)):
			temp=img[coor_noface[j,1]:coor_noface[j,3],coor_noface[j,0]:coor_noface[j,2]]
			noface[ctr_noface,:,:]=cv2.resize(temp, scalesize)
			ctr_noface+=1
	return face, noface
def sample_train_data(tr_pos, tr_neg, num_pos,num_neg, shuffle=False, rotate=True):
	a=np.arange(0,tr_neg.shape[0])
	np.random.shuffle(a)
	sample_neg=tr_neg[a[0:num_neg],:,:] # sample from negative images
	a=np.arange(0,tr_pos.shape[0])
	np.random.shuffle(a)
	sample_pos=tr_pos[a[0:num_pos],:,:]
	
	if rotate: # add 180 degree rotated image into training samples
		b=np.flip(sample_neg,axis=0)
		b=np.flip(b,axis=1)
		sample_neg=np.vstack((sample_neg,b))
		b=np.flip(sample_pos,axis=0)
		b=np.flip(b,axis=1)
		sample_pos=np.vstack((sample_pos,b))
	
	tr_y=np.concatenate((np.ones((sample_pos.shape[0],)),np.zeros((sample_neg.shape[0],))))
	tr_x=np.concatenate((sample_pos,sample_neg),axis=0)
	if shuffle:
		a=np.arange(0,tr_x.shape[0])
		np.random.shuffle(a)
		tr_x=tr_x[a,:,:]
		tr_y=tr_y[a]
	return tr_x, tr_y
def plot_training_image(tr_pos, tr_neg):
	rt=3
	cter=1
	idx_pos=np.random.randint(0,len(tr_pos),rt)
	idx_neg=np.random.randint(0,len(tr_neg),rt)
	for r in range(rt):
		plt.subplot(rt, 2, cter)
		plt.imshow(tr_pos[idx_pos[r],:,:])
		cter+=1
		plt.subplot(rt, 2, cter)
		plt.imshow(tr_neg[idx_neg[r],:,:])
		cter+=1
def read_testimage(path, style="Gray"):
	imgdic={}
	for i in range(1000):
		fname=str(i+1)+".jpg"
		fpath=path+"/"+fname
		if style=="Gray":
			img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
		elif style=="RGB":
			img = cv2.imread(fpath, cv2.IMREAD_COLOR)
		if not img.dtype == np.uint8:
			pass
		imgdic[fname]=img
	return imgdic

# Class 
class simple_classifier:
		"""
		x is all feature values on image(s)
		"""
		def __init__(self,th_arr=np.array([0]),polar_arr=np.array([1])):
				self.c_th=th_arr
				self.c_polar=polar_arr
				pass        
		def predict(self, x):
				y=(np.subtract(x, self.c_th)*self.c_polar)>0
				y=y.astype(float)
				return y
class strong_classifier:
		def __init__(self, feat_idx, alpha, th_arr, polar_arr, weak_train=True):
				"""
				feat_idx and alpha have same length
				th_arr, and polar_arr are for all features
				"""
				self.idx=feat_idx
				self.alpha=alpha
				self.th=th_arr
				self.polar=polar_arr
				self.weak_train=weak_train
		def predict(self, x):
				if self.weak_train:
						weakcf=simple_classifier(self.th[self.idx], self.polar[self.idx]) # create weak cf using given feature and threshold
				else:
						weakcf=simple_classifier() # create weak cf using 0 threshold, works for any size of feature
				pred=weakcf.predict(x) # size [num of images, num of features]
				return (pred.dot(self.alpha)-0.5*np.sum(self.alpha)>=0).astype(int)
class Haarfeature:
		"""
		Haarfeature
		(1,2) |white|gray|
		(2,1) gray
					white
		(1,3) |white|gray|white|
		(3,1) white
					gray
					white
		(2,2) |white|gray |
					|gray |white|
		feature value = sum of gray-sum of white
		Descrimptior: [v,u,v',u',feature type,featuresize/basesize]
		"""
		TWO_H=0
		TWO_V=1
		THREE_H=2
		THREE_V=3
		FOUR_S=4
		ALL=5
		FeatureShape=[(1,2),(2,1),(1,3),(3,1),(2,2)]
		Descriptor=np.array([]) # [v,u,v',u',feature type,featuresize/basesize]
		def __init__(self, winshape,min_fsize=1,max_fsize=None,stride="Adaptive"): # nrow,ncol are size of window in which feature is selected
				"""
				min_fsize and max_fize are times of the base features,not the pixel size
				""" 
				nrow, ncol=winshape
				count=0 # total number of features
				n=len(self.FeatureShape)
				if stride=="Adaptive":
						stride_scale=5
				if max_fsize is None:
						max_fsize=math.inf
				for fs in range(n):
						feature=self.FeatureShape[fs]
						v_len_max=min(int(nrow/feature[0]),max_fsize)
						for i in range(min_fsize, v_len_max+1):
								v_len=int(i*feature[0])
								u_len=int(v_len/feature[0]*feature[1])
								if u_len>ncol:
										continue
								if stride=="Adaptive":
										st_v=int(v_len/stride_scale)
										st_u=int(u_len/stride_scale)
								else:
										st_v=stride
										st_u=stride
								count+=(int((nrow-v_len)/st_v)+1)*(int((ncol-u_len)/st_u)+1)
								#                 count+=(nrow-v_len+1)*(ncol-u_len+1)
				print("Number of features",count)
				self.Descriptor=np.zeros((count, 6), dtype=np.int)
				index=0
				for fs in range(n):
						feature=self.FeatureShape[fs]
						v_len_max=min(int(nrow/feature[0]),max_fsize)
						for i in range(min_fsize, v_len_max+1):
								v_len=int(i*feature[0])# feature[0] row, feature[1] column
								u_len=int(v_len/feature[0]*feature[1])
								if u_len>ncol:
										continue
								if stride=="Adaptive":
										st_v=int(v_len/stride_scale)
										st_u=int(u_len/stride_scale)
								else:
										st_v=stride
										st_u=stride
								for v in range(0, int((nrow-v_len)/st_v)+1):
										for u in range(0, int((ncol-u_len)/st_u)+1):
												self.Descriptor[index]=[v, u, v+v_len-1, u+u_len-1, fs, i]
												index+=1
		
		def get_TWO_H(self, LU,RB,LB,RU,img_integral):
				MU=np.array([LU[:,0],(LU[:,1]+RU[:,1])/2],dtype=int).T # find middle point, (selected feature size, 2)
				MB=np.array([LB[:,0],(LB[:,1]+RB[:,1])/2],dtype=int).T
				corners=[LU,RB,LB,RU,MU,MB]
				areas=self.cal_area(corners, img_integral)
				return areas[:,1]+areas[:,2]+2*areas[:,4]-2*areas[:,5]-areas[:,0]-areas[:,3]
		def get_TWO_V(self, LU,RB,LB,RU,img_integral):
				ML=np.array([(LU[:,0]+LB[:,0])/2,LU[:,1]],dtype=int).T # find middle point
				MR=np.array([(RU[:,0]+RB[:,0])/2,RU[:,1]],dtype=int).T
				corners=[LU,RB,LB,RU,ML,MR]
				areas=self.cal_area(corners, img_integral)
				return areas[:,0]+areas[:,2]+2*areas[:,5]-2*areas[:,4]-areas[:,3]-areas[:,1]
		def get_THREE_H(self, LU,RB,LB,RU,img_integral):
				MU1=np.array([LU[:,0],(2*LU[:,1]+RU[:,1])/3],dtype=int).T # find first tri-section point
				MU2=np.array([LU[:,0],(LU[:,1]+2*RU[:,1])/3],dtype=int).T # find second tri-section point
				MB1=np.array([LB[:,0],(2*LB[:,1]+RB[:,1])/3],dtype=int).T
				MB2=np.array([LB[:,0],(LB[:,1]+2*RB[:,1])/3],dtype=int).T
#         print(LU,'\n',MU1,'\n',MU2,'\n',RU)
				corners=[LU,RB,LB,RU,MU1,MU2,MB1,MB2]
				areas=self.cal_area(corners, img_integral)
				return areas[:,2]+areas[:,3]+2*areas[:,4]+2*areas[:,7]-areas[:,0]-areas[:,1]-2*areas[:,5]-2*areas[:,6]
		def get_THREE_V(self, LU,RB,LB,RU,img_integral):
				ML1=np.array([(2*LU[:,0]+LB[:,0])/3,LU[:,1]],dtype=int).T # first tri-section point in vertical direction
				ML2=np.array([(LU[:,0]+2*LB[:,0])/3,LU[:,1]],dtype=int).T
				MR1=np.array([(2*RU[:,0]+RB[:,0])/3,RU[:,1]],dtype=int).T
				MR2=np.array([(RU[:,0]+2*RB[:,0])/3,RU[:,1]],dtype=int).T
				corners=[LU,RB,LB,RU,ML1,ML2,MR1,MR2]
				areas=self.cal_area(corners, img_integral)
				return areas[:,2]+areas[:,3]+2*areas[:,4]+2*areas[:,7]-areas[:,0]-areas[:,1]-2*areas[:,5]-2*areas[:,6]
		def get_FOUR_S(self, LU,RB,LB,RU,img_integral):
				MU=np.array([LU[:,0],(LU[:,1]+RU[:,1])/2],dtype=int).T
				MB=np.array([LB[:,0],(LB[:,1]+RB[:,1])/2],dtype=int).T
				ML=np.array([(LU[:,0]+LB[:,0])/2,LU[:,1]],dtype=int).T
				MR=np.array([(RU[:,0]+RB[:,0])/2,RU[:,1]],dtype=int).T
				MO=np.array([(LU[:,0]+LB[:,0])/2,(LU[:,1]+RU[:,1])/2],dtype=int).T # Middle point in square
				corners=[LU,RB,LB,RU,MU,MB,ML,MR,MO]
				areas=self.cal_area(corners, img_integral)
				return 2*(areas[:,4]+areas[:,5]+areas[:,6]+areas[:,7])-(areas[:,0]+areas[:,1]+areas[:,2]+areas[:,3])-4*areas[:,8] 
		def cal_area(self, corners, img_integral):
				areas=np.zeros((corners[0].shape[0],len(corners)))
				rnum,cnum=img_integral.shape
				for i in range(len(corners)):
						corner=corners[i] # each corner has shape (feature num, 2)
#             tempidx=np.where((corner[:,0]>=0)&(corner[:,1]>=0)&(corner[:,0]<rnum)&(corner[:,1]<cnum))[0] 
						tempidx=np.where((corner[:,0]>=0)&(corner[:,1]>=0))[0] 
						# if coor is padding, automatically zero area
						areas[tempidx,i]=img_integral[corner[tempidx,0],corner[tempidx,1]]
#             tempidx=np.where((corner[:,0]>=rnum)&(corner[:,1]<cnum))[0] # row out of bound
#             areas[tempidx,i]=img_integral[-1,corner[tempidx,1]]
#             tempidx=np.where((corner[:,0]<rnum)&(corner[:,1]>=cnum))[0] # column out of bound
#             areas[tempidx,i]=img_integral[corner[tempidx,0],-1]
#             tempidx=np.where((corner[:,0]>=rnum)&(corner[:,1]>=cnum))[0]  # both out of bound
#             areas[tempidx,i]=img_integral[-1,-1]
						
				return areas
		def get_feature_value(self, img_integral, fsize=None, ftype=None, givenindex=None, scale=None):
				"""
				# Parameters
				fsize:
						The multiply number of base feature size. Example, (1,2)*2=(2,4)
				ftype:
						Type of feature, a list of integer from 0 to 4.
				# return
						Haar-like feature value
				"""
				if ftype!=None:
						logbit_1=np.zeros(self.Descriptor.shape[0],dtype=bool)
						for i in range(len(ftype)):
								logbit_1=logbit_1 | (self.Descriptor[:,4]==ftype[i])
				else:
						logbit_1=np.ones(self.Descriptor.shape[0],dtype=bool)
				if fsize!=None:
						logbit_2=np.zeros(self.Descriptor.shape[0],dtype=bool)
						for i in range(len(fsize)):
								logbit_2=logbit_2 | (self.Descriptor[:,5]==fsize[i])
				else:
						logbit_2=np.ones(self.Descriptor.shape[0],dtype=bool)
				
				rbound=img_integral.shape[0]
				cbound=img_integral.shape[1]
				logbit_3=(self.Descriptor[:,0]>=0)&(self.Descriptor[:,1]>=0)&(self.Descriptor[:,2]<rbound)&(self.Descriptor[:,3]<cbound)
				logbit=logbit_1 & logbit_2 & logbit_3
				if givenindex is None:
						index=np.where(logbit)[0] # record the index of useful feature in descriptor
				else:
						index=givenindex
				
				if scale is None:
						feature=self.Descriptor[index,:]
				else:
						feature=np.copy(self.Descriptor[index,:])
						feature[:,0:4]=(feature[:,0:4]*scale).astype(int)
#             modify=feature[:,2]>=img_integral.shape[0]
#             feature[np.where(modify)[0], 2]=img_integral.shape[0]
#             modify=feature[:,3]>=img_integral.shape[1]
#             feature[np.where(modify)[0], 3]=img_integral.shape[1]
						
				LU=np.array([feature[:,0]-1,feature[:,1]-1]).T # left up corner coordinates. -1 mean left padding one
				RB=feature[:,2:4] # right bottom corner
				LB=np.array([feature[:,2],feature[:,1]-1]).T # left bottom corner. [[x1,y1],[x2,y2]...],(feature size, 2)
				RU=np.array([feature[:,0]-1,feature[:,3]]).T # right up corner
				res=np.zeros((feature.shape[0],))
				for t in range(self.ALL):
						inte_idx=np.where(feature[:,4]==t)[0] # index for integral image
						if t==self.TWO_H:
								temp=self.get_TWO_H(LU[inte_idx,:],RB[inte_idx,:],LB[inte_idx,:],RU[inte_idx,:],img_integral)
						if t==self.TWO_V:
								temp=self.get_TWO_V(LU[inte_idx,:],RB[inte_idx,:],LB[inte_idx,:],RU[inte_idx,:],img_integral)
						if t==self.THREE_H:
								temp=self.get_THREE_H(LU[inte_idx,:],RB[inte_idx,:],LB[inte_idx,:],RU[inte_idx,:],img_integral)
						if t==self.THREE_V:
								temp=self.get_THREE_V(LU[inte_idx,:],RB[inte_idx,:],LB[inte_idx,:],RU[inte_idx,:],img_integral)
						if t==self.FOUR_S:
								temp=self.get_FOUR_S(LU[inte_idx,:],RB[inte_idx,:],LB[inte_idx,:],RU[inte_idx,:],img_integral)
						res[inte_idx]=temp
						
				return res,index


# Adaboost
def cal_feature_value(integral_images,feat):
	imgidx=list(images.keys())
	feat_arr=np.zeros((integral_images.shape[0],feat.Descriptor.shape[0]))
	print("Start calculating all feature values for each image")
	for i in tqdm(range(integral_images.shape[0])):
		inteimg=integral_images[i,:,:]
		featval,idx_feat=feat.get_feature_value(inteimg, fsize=None, ftype=None) # for each image
		feat_arr[i,idx_feat]=featval
	return feat_arr
def train_simple_classifier(featval, y_true):
	# featval: [num of imgs, num of features]
	total_pos=np.where(y_true==1)[0].shape[0]
	total_neg=y_true.shape[0]-total_pos
	
	n_img=featval.shape[0]
	th_arr=np.zeros((featval.shape[1]))
	polar_arr=np.zeros((featval.shape[1]))
	
	for i in tqdm(range(featval.shape[1])):
		idx=np.argsort(featval[:,i])    # ith feature's index on image 
		pos_seen, neg_seen=0, 0
		right_max=-1
		for j in range(n_img):
			a=np.array([neg_seen+total_pos-pos_seen, pos_seen+total_neg-neg_seen]) # left neg right pos or right neg left pos
			a_idx=np.argmax(np.array([neg_seen+total_pos-pos_seen, pos_seen+total_neg-neg_seen]))
			right=a[a_idx]
			if right>right_max:
				right_max=right
				th_arr[i]=featval[idx[j],i] # ith feature's threshold is current feature value on this image
			if a_idx==0:
				polar_arr[i]=1 # >th is pos, so left neg and right pos
			else:
				polar_arr[i]=-1 # <th is pos, so left pos and right neg
			if y_true[idx][j]==1: # sorted label index at j
				pos_seen+=1
			else:
				neg_seen+=1
	return th_arr, polar_arr

def Adaboost(tr_x, tr_y, error, integral_images, feat, num_feature=-1):
	
#     global error
	num_pos=tr_x.shape[0]/2
	num_neg=tr_x.shape[0]-num_pos
	wt=1/2/num_pos*tr_y+1/2/num_neg*(1-tr_y) # initial weight on images for each feature
	num_feature=feat.Descriptor.shape[0] if num_feature==-1 else num_feature
	
	print("weights shape: ",wt.shape)
	print("Calculating all prediction errors that equal one")
	one_list=[]
	for i in tqdm(range(error.shape[1])): # loop features
		one_list.append(np.where(error[:,i]==1)[0])
	print("Start Adaboost")
	feat_idx=np.arange(feat.Descriptor.shape[0])
	best_feat_list=[]
	alpha=[]
#     alpha=np.zeros((feat.Descriptor.shape[0],))
	for i in tqdm(range(num_feature)):
		e_weight=np.zeros((len(feat_idx),))
		for j in range(len(feat_idx)):
			e_weight[j]=np.sum(wt[one_list[feat_idx[j]]])# only cal weighted sum with error==1
		to_feat_idx=np.argmin(e_weight)
		best_feat_idx=feat_idx[to_feat_idx]
		min_e_weight=e_weight[best_feat_idx]
		if min_e_weight==1:
			min_e_weight=1-1e-6
		elif min_e_weight==0:
			min_e_weight=1e-6
#         beta=np.sqrt(min_e_weight/(1-min_e_weight))
		beta=min_e_weight/(1-min_e_weight)
		idx=np.where(error[:,best_feat_idx]==0)[0] # find correct prediction
		wt[idx]=wt[idx]*beta # update weight
		wt=wt/np.sum(wt) # normalize weights
		alpha.append(np.log(1/beta))
		best_feat_list.append(best_feat_idx)
#         feat_idx=np.delete(feat_idx,to_feat_idx)# remove seleted feature
	print("Finish training.")
	return np.array(alpha),wt,np.array(best_feat_list)
def evaluate(image,label, feat, strongclf, integral_images=None, feat_value_all=None):
	feat_idx=strongclf.idx
	if integral_images is None:
		integral_images=cal_integral_image_dict(image)
	if feat_value_all is None:
		feat_arr=np.zeros((integral_images.shape[0],feat_idx.shape[0]))
		for i in tqdm(range(integral_images.shape[0])):
			inteimg=integral_images[i,:,:]
			featval,_=feat.get_feature_value(inteimg, fsize=None, ftype=None, givenindex=feat_idx) # for each image
			feat_arr[i,:]=featval
		pred=strongclf.predict(feat_arr)
	else:
		pred=strongclf.predict(feat_value_all[:,feat_idx])
	tp=np.where(((pred==1)&(label==1)))[0].shape[0]
	fp=np.where(((pred==1)&(label==0)))[0].shape[0]
	tn=np.where(((pred==0)&(label==0)))[0].shape[0]
	fn=np.where(((pred==0)&(label==1)))[0].shape[0]
	return tp,fp,tn,fn

# Detection
def detect(feat,strongclf, detect_win, image_target,f_shift=0.8, f_scale=1.25, nscale=4,scaleth=True,casc=True): # scan image and predict using multiscale detector
	integral_image=cal_integral_image_dict(np.expand_dims(image_target, axis=0))
	feat_idx=strongclf.idx
	im_r, im_c=image_target.shape
	print("Image shape and detect window",image_target.shape, detect_win)    
	win_r, win_c=detect_win
	cur_scale=1
#     th_=np.copy(th)
	
	if f_scale==1:
		maxscale=0
	else:
		maxscale=min(int(np.log(im_r/nscale/win_r)/np.log(f_scale)),int(np.log(im_c/nscale/win_c)/np.log(f_scale)))
	print("Maximum scale: ",maxscale)
	res=[]
	for k in range(maxscale):#range(maxscale+1)
		dr, dc=int(win_r*(f_scale**k)), int(win_c*(f_scale**k)) # update scaled window
		cur_scale=f_scale**k # update scale factor to calcualte feature coordinate
		if scaleth:
			th_=strongclf.th*(f_scale**k)
		else:
			th_=strongclf.th
		sc=int(dc*f_shift) # stride is a scale of window size
		sr=int(dr*f_shift)
#         sc, sr = int(dc),int(dr) # Define stride
		
		nr=int((im_r-dr)/sr)+1
		nc=int((im_c-dc)/sc)+1
		
		res_temp=[]
#         for i in tqdm_notebook(range(0, nr)): # shift the window with stride sc and sr
		for i in range(0, nr):
			for j in range(0, nc):
				v=i*sr
				u=j*sc
				if i==nr:
					v=im_r-dr
				if j==nc:
					u=im_c-dc
				if u+dc>im_c or v+dr>im_r:
						continue
#                 print(v,v+dr-1,u,u+dc-1)
				featval, _=feat.get_feature_value(integral_image[0][v:v+dr, u:u+dc], 
											   fsize=None, ftype=None, givenindex=feat_idx,scale=cur_scale)
				if casc:  
					pred=small_cascade(strongclf, featval)
				else:
					pred=strongclf.predict(featval)
					
				
		
				if pred==1:
					res_temp.append([u,v,u+dc-1,v+dr-1])
		res.append(np.array(res_temp))
		
	return res
def small_cascade(strongclf, featval): # mini cascade detector embedded in strong classifier detection
	n=strongclf.alpha.shape[0]
	cascade=[]
	i=1
	while 2**i<=n:
		cascade.append(2**i)
		i=i+1
#     i=1
#     while i<=n:
#         cascade.append(i)
#         i=i+20
#     print("size of cascade: ", len(cascade))
	for j in range(len(cascade)):
		nl=cascade[j]
		strcls_t=strong_classifier(strongclf.idx[0:nl], strongclf.alpha[0:nl], strongclf.th, strongclf.polar, weak_train=True)
		if strcls_t.predict(featval[0:nl])==0:
			return 0
	return 1
def detect_batch(imgs):# create bounding box for each image
	bound={}
	counter=0
	for k, v in imgs.items():
		print(str(counter)+"th: ")
		detected=detect(feat, strongclf, (50,40), v, f_shift=1, f_scale=1.25,nscale=2,scaleth=True,casc=True)
#         merged_final=mergerect(detected, ningroup=1, eps=0.2)
		bound[k]=detected # numpy
		counter+=1
	return detected
def create_json(fname,bound): # create json from bounding box
	tojson={}
	counter=0
	with open("results.json","w") as f:
		for k, v in bound.items():
			for r in v:
				x=r.tolist()
				x[2]=x[2]-x[0]+1
				x[3]=x[3]-x[1]+1
				tojson={"iname": k, "bbox":x}
				json.dump(tojson, f, indent=2)
				counter+=1
	print(counter)
	f.close()
def merge_batch(bound): # merge bounding box
	res={}
	for k, v in bound.items():
		merged_final=mergerect(v, ningroup=1, eps=0.2)
		res[k]=merged_final
	return res
	
	
# Main script

# Read image and target region
cwd=os.getcwd()
index_path="FDDB/FDDB-folds/" 
img_path="FDDB/originalPics/"
print("Load training image from folder")
region=read_target_region(cwd+"/"+index_path)
image_index=list(region.keys())
images=read_image(image_index, img_path, style="Gray")
# Calculte size of face image
facebound,facebound_imgid=cal_face_boundary(region)
print("The minimum face size: ",(facebound[2], facebound[0]))
print("Image ID with minimum face size: ", facebound_imgid[0])
print("The maximum face size: ",(facebound[3], facebound[1]))
print("Image ID with maximum face size: ", facebound_imgid[1])
# Read test images
print("Load testing image from folders")
test_path=os.getcwd()+"/test_images_1000"
images_test=read_testimage(test_path, style="Gray")

# Set control parameter
num_tr=2000 # sample number from positive or negative image sets for training
num_va=500 # sample number for testing
load_data=False # Determine whether to load data for convienence or compute from scratch
weaktrain=True # Determine whether to use trained threhold for weak classifiers
scale_size=(40,50) # Detecting window, also window for haar feature cacluation
noface_size=(80,80) # window to get non-face image patch
num_feat_select=350 # the number of feature selected for strong classifier
cwd=os.getcwd()

# Load or create datasets
#if load_data:
#	tr_pos=np.load(cwd+"/result/face.npy")
#	tr_neg=np.load(cwd+"/result/noface.npy")
#	tr_x=np.load(cwd+"/result/train_img.npy")
#	tr_y=np.load(cwd+"/result/train_lable.npy")
#	va_x=np.load(cwd+"/result/test_img.npy")
#	va_y=np.load(cwd+"/result/test_lable.npy")
#else:
tr_pos,tr_neg=get_train_data(images, region,win_noface=noface_size,scalesize=scale_size)
tr_x, tr_y =sample_train_data(tr_pos, tr_neg, num_tr, num_tr*2, shuffle=False, rotate=False)
va_x, va_y =sample_train_data(tr_pos, tr_neg, num_va, num_va*2, shuffle=False, rotate=False)
#	np.save(cwd+"/result/face.npy",tr_pos)
#	np.save(cwd+"/result/noface.npy",tr_neg)
#	np.save(cwd+"/result/train_img.npy",tr_x)
#	np.save(cwd+"/result/train_lable",tr_y)
#	np.save(cwd+"/result/test_img.npy",va_x)
#	np.save(cwd+"/result/test_lable.npy",va_y)
	
print("Size of images and labels in training set: ",tr_x.shape, tr_y.shape)
print("Size of images and labels in testing set: ", va_x.shape, va_y.shape)
# Calculate feature 
print("Calculate haar feature coordinates in detecting window")
feat=Haarfeature(scale_size,min_fsize=1,max_fsize=None,stride=1) # feature definition

print("Create integral image, calculate feature values on each image")
integral_images=cal_integral_image_dict(tr_x) # calculate integral image for training set
featval=cal_feature_value(integral_images,feat) # calculate feature value on integral image

if load_data: # load or calculate best feature threshold and polarity
	print("Load threshold for weak classifiers. It is easy.")
	th_best=np.load(cwd+"/result/th_cls.npy")
	polar_best=np.load(cwd+"/result/polar_cls.npy")
#	integral_images=np.load(cwd+"/result/integral_image.npy")
#	featval=np.load(cwd+"/result/featval.npy")
	
else:
#		np.save(cwd+"/result/integral_image.npy",integral_images)
	print("Start training threshold for weak classifiers, please be patient..")
	th_best, polar_best=train_simple_classifier(featval, tr_y)
#	np.save(cwd+"/result/th_cls.npy",th_best)
#	np.save(cwd+"/result/polar_cls.npy",polar_best)
#	np.save(cwd+"/result/featval.npy",featval)
#	np.save(cwd+"/result/integral_image.npy",integral_images)

# Build weak classifiers
if weaktrain: # if use best parameter or default
	simple_cls=simple_classifier(th_arr=th_best, polar_arr=polar_best)
else:
	simple_cls=simple_classifier()
# Load or create prediction error for Adaboost
#if load_data: # load
#	error=np.load(cwd+"/result/error.npy")
#else:
print("Calculate prediction error of each feature on each image")
error=np.abs(simple_cls.predict(featval)-tr_y.reshape(-1,1)).astype(bool)
#np.save(cwd+"/result/error.npy",error)

# Adaboost
alpha, weight,feat_select=Adaboost(tr_x, tr_y, error, integral_images,feat,num_feature=num_feat_select)
testfeat=np.arange(0,len(feat_select)) # manually choose feature
strongclf=strong_classifier(feat_select[testfeat], alpha, th_best, polar_best, weak_train=weaktrain)# build strong classifer
print("Shape of alpha and selected feature",alpha.shape, feat_select.shape)
# Evaluation on training data
tp,fp,tn,fn=evaluate(tr_x, tr_y, feat, strongclf, integral_images=integral_images,feat_value_all=featval)
print("True postive, false positive, true negative, false negative: ",tp,fp,tn,fn)
print("Precision: ", tp/(tp+fp))
print("Recall: ",tp/(tp+fn))
print("Accuracy: ", (tp+tn)/(tp+tn+fp+fn))
# Evaluation on testing data
tp,fp,tn,fn=evaluate(va_x, va_y, feat, strongclf, integral_images=None,feat_value_all=None)
print("True postive, false positive, true negative, false negative: ",tp,fp,tn,fn)
print("Precision: ", tp/(tp+fp))
print("Recall: ",tp/(tp+fn))
print("Accuracy: ", (tp+tn)/(tp+tn+fp+fn))

# Detect
print("Starte detecting face in image")
bound_nomerg=detect_batch(images_test) # Find bounding box for test images without merging
bound=merge_batch(bound_nomerg, eps=0.4) # batch merging bounding boxes
create_json("result_json.json",bound)