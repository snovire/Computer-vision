#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import os

import random
from sklearn.metrics.pairwise import cosine_similarity
import sys



def read_image(img_path, style="Gray",show=False ):
    """Reads an image into memory 
    """
    if style=="Gray":
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    elif style=="RGB":
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.imread(img_path)

    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)
    return img
def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()
def read_batch(paths, sty="Gray"):
    imgs=[]
    for i in range(len(paths)):
        img=read_image(paths[i], style=sty)
        imgs.append(img)
        print(str(i)+"th image shape: ",img.shape)
    return imgs
def feature_detect(imgs, num_feature=0, contrast_th=0.04):
    sift=cv2.xfeatures2d.SIFT_create(nfeatures=num_feature, contrastThreshold=contrast_th)
    kp_list=[]
    des_list=[]
#     img_output=[]
    for i in range(len(imgs)):
        kp, des= sift.detectAndCompute(imgs[i],None)
        kp_list.append(kp)
        des_list.append(des)
#         img_output.append(cv2.drawKeypoints(imgs[i],kp,np.array([])))
    return kp_list, des_list
def homography_trans(coor, H):
    u=coor[0] # column number
    v=coor[1] # row number
    vec=np.array([u,v,1]).reshape(-1,1)
    return H.dot(vec)



def refine_keypoint(keypoints,img_size,tol):
    tol_coor=tolerant_area(tol)
    kp_neighbor_list=-1*np.ones(img_size,dtype=np.int)
    for i in range(len(keypoints)):
        u=int(keypoints[i].pt[0]) # column number
        v=int(keypoints[i].pt[1])
        neighbor_ok=True
        for j in range(tol_coor.shape[0]):
            r=tol_coor[j][0]+v
            c=tol_coor[j][1]+u
            if (r>=0) and (r<img_size[0]) and (c>=0) and (c<img_size[1]):
                if kp_neighbor_list[r][c]!=-1:
                    neighbor_ok=False
                    break
        if neighbor_ok and kp_neighbor_list[v][u]==-1:
            kp_neighbor_list[v][u]=i
    add_list=[]
    for i in range(len(keypoints)):
        u=int(keypoints[i].pt[0]) # column number
        v=int(keypoints[i].pt[1]) # row number
        if kp_neighbor_list[v][u]!=-1:
            add_list.append(kp_neighbor_list[v][u])
            kp_neighbor_list[v][u]=-1
    return add_list
def tolerant_area(tol):
    arr=[]
    for i in range(-tol,tol+1):
        for j in range(-tol,tol+1):
            if i!=0 or j!=0: # avoid centor point
                arr.append([i,j])
    arr=np.array(arr)
    return arr
def get_refined_list(add_list,keypoints, descriptors):
    # Get refined keypoints
    add_sort=sorted(add_list)
    keypoints_new=[]
    descriptors_new=[]
    j=0
    i=0
    while i<len(keypoints) and j<len(add_sort):
        if i==add_sort[j]:
            keypoints_new.append(keypoints[i])
            descriptors_new.append(list(descriptors[i,:]))
            j+=1
        elif i>add_sort[j]:
            print("Add list error")
        i+=1
    while i<len(keypoints):
        keypoints_new.append(keypoints[i])
        descriptors_new.append(list(descriptors[i,:]))
        i+=1
    return keypoints_new, np.array(descriptors_new)
def refine_feature(kp_list,des_list, img_size, tol):
    kp_list_new=[]
    des_list_new=[]
    for i in range(len(kp_list)):
        rm_list=refine_keypoint(kp_list[i],img_size,tol)
        kp_new,des_new=get_refined_list(rm_list,kp_list[i],des_list[i])
        kp_list_new.append(kp_new)
        des_list_new.append(des_new)
    return kp_list_new, des_list_new
def draw(imgs, kp_list, rich_draw=False):
    img_output=[]
    for i in range(len(imgs)):
        if rich_draw:
            flg=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        else:
            flg=0
        img_output.append(cv2.drawKeypoints(imgs[i],kp_list[i],np.array([]),flags=flg))
    return img_output



def shuffle_partition(n_pick,n_data):
    all_index=np.arange(n_data)
    np.random.shuffle(all_index)
    index_pick=all_index[0:n_pick]
    index_test=all_index[n_pick:]
    return index_pick, index_test
def feature_pair(des1, des2):
    cos_sim=cosine_similarity(des1, des2)
#     print(cos_sim.shape)
    pairs=[]
    sims=[]
    for i in range(cos_sim.shape[0]):
        index=np.flip(np.argsort(cos_sim[i,:]))
#         print(cos_sim[i, index[0:4]])
        for j in range(cos_sim.shape[1]-1):
            if cos_sim[i,index[j+1]]<cos_sim[i,index[j]]*0.98:
                pairs.append([i,index[j]])
                sims.append(cos_sim[i,index[j]])
                break
    return np.array(pairs), np.array(sims)
def extract_point_coor(pair_feat,keypoints,keypoints_prime):
    kp=[]
    kp_prime=[]
    for i in range(pair_feat.shape[0]):
        index_kp=pair_feat[i,0]
        index_kp_prime=pair_feat[i,1]
        kp.append(keypoints[index_kp].pt)
        kp_prime.append(keypoints_prime[index_kp_prime].pt)
    return np.array(kp), np.array(kp_prime) # n*2 array
def cal_fit_error(H, kp, kp_prime): # kp=[[u,v],], is a n*2 array
    n=kp.shape[0]
    add=np.ones((1,n))
    X=np.vstack((kp.T,add))# 3*n_points
    X_p=np.vstack((kp_prime.T,add))
    delta=H.dot(X)-X_p
    sum_square_error=np.sum(np.square(delta),axis=0) # sum per column
#     print(np.max(delta[2,:]))
    return sum_square_error.reshape(-1)
def image_transform(H, coor, img, img_notrans):
    
    coor_trans=H.dot(coor).astype(int) # homogeneous coordinates in transformed image
#     print("Shape of cooridnates ",coor_trans.shape)
    row_min=min(np.min(coor_trans[1,:]),np.min(coor[1,:]))
    col_min=min(np.min(coor_trans[0,:]),np.min(coor[0,:]))
    row_max=max(np.max(coor_trans[1,:]),np.max(coor[1,:]))
    col_max=max(np.max(coor_trans[0,:]),np.max(coor[0,:]))

    delta_r=0-row_min # defined new coordinate system by shifting along row and column
    delta_c=0-col_min
    row_max=row_max+delta_r
    col_max=col_max+delta_c
#     print("Row coordinates bound: ",(row_min,row_max))
#     print("Column coordinates bound: ",(col_min,col_max))
    coor_trans[1,:]+=delta_r # H transformed coordinates in new system
    coor_trans[0,:]+=delta_c
    img_out=np.zeros((row_max+1,col_max+1)) # H transformed image in new coordinate system
    img_out_notrans=np.zeros((row_max+1,col_max+1)) # No H transformed image in new coordinate system
    img_out_notrans[delta_r:delta_r+img_notrans.shape[0],delta_c:delta_c+img_notrans.shape[1]]=img_notrans
    img_pxnum=np.zeros(img_out.shape)
    
    img_out[coor_trans[1,:],coor_trans[0,:]]=img[coor[1,:],coor[0,:]] # value of repeat key will be overwritten
    img_pxnum[coor_trans[1,:],coor_trans[0,:]]=1
    r_zero, c_zero=np.where(img_pxnum==0)
    img_pad=np.vstack((np.zeros((1,img_out.shape[1])),img_out,np.zeros((1,img_out.shape[1]))))
    img_pad=np.hstack((np.zeros((img_out.shape[0]+2,1)),img_pad,np.zeros((img_out.shape[0]+2,1))))
    img_out[r_zero,c_zero]=img_pad[r_zero+1,c_zero+2]
#     img_out[r_zero,c_zero]=(img_pad[r_zero,c_zero]+img_pad[r_zero,c_zero+1]+img_pad[r_zero,c_zero+2]+
#                             img_pad[r_zero+1,c_zero]+img_pad[r_zero+1,c_zero+2]+
#                             img_pad[r_zero+2,c_zero]+img_pad[r_zero+2,c_zero+1]+img_pad[r_zero+2,c_zero+2])/8
    
    idx_zero=np.where(img_pxnum==0)
    img_pxnum[idx_zero]=1
    return np.uint8(img_out/img_pxnum), np.uint8(img_out_notrans)
def blend_image(img1, img2, ratio1):
    # img1 and img2 must have the same shape
    img_out=np.zeros(img1.shape)
    for i in range(img_out.shape[2]):
        a=img1[:,:,i].reshape(1,-1)[0]
        b=img2[:,:,i].reshape(1,-1)[0]
        idx_equal_b=np.where(np.logical_and(a==0,b>0))
        idx_equal_a=np.where(b==0)
        idx_blend=np.where(np.logical_and(a>0,b>0))
        c=np.zeros(a.shape)
        c[idx_equal_a]=a[idx_equal_a]
        c[idx_equal_b]=b[idx_equal_b]
        c[idx_blend]=ratio1*a[idx_blend]+(1-ratio1)*b[idx_blend]
        img_out[:,:,i]=c.reshape(img_out.shape[0],img_out.shape[1])
    return np.uint8(img_out)
def cal_homo_coor(img_size):
    r=img_size[0]
    c=img_size[1]
    first=np.arange(0,c)
    first=np.tile(first,(1,r))
    second=np.arange(0,r)
    second=np.tile(second,(c,1)).transpose().reshape(1,-1)
    third=np.ones((r*c,))
    coor=np.vstack((first, second, third))
    return coor.astype(int)
def est_homography(kp, kp_prime):# kp=[[u,v],]
    try:
        A=[]
        b=[]
        for i in range(kp.shape[0]):
            u=kp[i][0] # column direction(max 3888)
            v=kp[i][1] # row direction(max 2592)
            u_p=kp_prime[i][0]
            v_p=kp_prime[i][1]
            A.append([u,v,1,0,0,0,0,0])
            A.append([0,0,0,u,v,1,0,0])
            A.append([0,0,0,0,0,0,u,v])

    #         A.append([u,v,1,0,0,0,0,0,0])
    #         A.append([0,0,0,u,v,1,0,0,0])
    #         A.append([0,0,0,0,0,0,u,v,1])

            b.append(u_p)
            b.append(v_p)
            b.append(0)
    #         b.append(1)

        A=np.array(A)
        b=np.array(b).reshape(-1,1)
        H=np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        H=np.vstack((H,np.array(1))).reshape(3,3)
#         H=H.reshape(3,3)
#         H=H/H[2,2]
    except np.linalg.LinAlgError as err:
#         print("Error: {0}. Use identity matrix instead".format(err)) # print error
        H=np.identity(3)
    return H
def ransac(keypoints,keypoints_prime,descriptors,descriptors_prime,itr=1,th_num=10,tol_pixel=3,num_sample=3):
    pair_feat, sim_feat=feature_pair(descriptors, descriptors_prime)
    print("The dimmention of paired keypoints in two images: ", pair_feat.shape)
    kp_all,kp_all_prime=extract_point_coor(pair_feat,keypoints,keypoints_prime)
    err_best=np.inf
    n_inliner_best=-1
    H_best=None
    th_err=1.1*(tol_pixel**2)
#     print(kp_all.shape, kp_all_prime.shape)
    for i in range(itr):
        index_fit, index_test=shuffle_partition(num_sample,kp_all.shape[0])# index in kp_all,kp_all_prime
        kp_fit=kp_all[index_fit,:]
        kp_fit_prime=kp_all_prime[index_fit,:]
        kp_test=kp_all[index_test,:]
        kp_test_prime=kp_all_prime[index_test,:]
        H=est_homography(kp_fit,kp_fit_prime)
        test_error=cal_fit_error(H, kp_test, kp_test_prime) # number of test points
        add_index=index_test[test_error<th_err] # find inliners in test points, admit fit points are inliners.
        add_inliner=kp_all[add_index,:]
        add_inliner_prime=kp_all_prime[add_index,:]
#         print(np.mean(test_error))
#         print("First fit error: ",cal_fit_error(H, kp_fit, kp_fit_prime))
        if add_inliner.shape[0]>th_num:
            numofadd=num_sample
            index_sort=np.argsort(test_error)
            idx=index_test[index_sort[0:numofadd]]
            new_kp=np.vstack((kp_fit,kp_all[idx,:]))
            new_kp_prime=np.vstack((kp_fit_prime,kp_all_prime[idx,:]))
            
            new_kp=kp_all[idx,:]
            new_kp_prime=kp_all_prime[idx,:]
            
#             new_kp=np.vstack((kp_fit,add_inliner))
#             new_kp_prime=np.vstack((kp_fit_prime,add_inliner_prime))
#             # random select again
#             ns=np.arange(new_kp.shape[0])
#             np.random.shuffle(ns)
#             new_kp_fit=new_kp[ns[0:num_sample],:]
#             new_kp_fit_prime=new_kp_prime[ns[0:num_sample],:]
#             # pick up several in order
#             new_kp_fit=add_inliner[0:num_sample,:]
#             new_kp_fit_prime=add_inliner_prime[0:num_sample,:]
            # pick up all
            new_kp_fit=new_kp
            new_kp_fit_prime=new_kp_prime
            
            H_new=est_homography(new_kp_fit, new_kp_fit_prime)
            error_new=cal_fit_error(H, kp_all, kp_all_prime)
            n_inliner=np.where(error_new<th_err)[0].shape[0]
            if n_inliner>n_inliner_best:
                n_inliner_best=n_inliner
                H_best=H_new
                inliner_index_best=np.hstack((index_fit, add_index))
#             err=np.mean(error_new) # a scalar
#             if err<err_best:
#                 err_best=err
#                 H_best=H_new
#                 inliner_index_best=np.hstack((index_fit, add_index))
    if H_best is None:
        raise ValueError("Insufficient inliner counts")
    print("number of inliners: ",n_inliner_best)
    return inliner_index_best, H_best


# Load image:

datapath=sys.argv[1]
img_names=os.listdir(datapath)
filename=[]
for i in range(len(img_names)):
    nsplit=img_names[i].split(".")
    if nsplit[-1]=="jpg":
        filename.append(img_names[i])
filename=sorted(filename)
    
#cwd=os.getcwd()
#path_part=["/data/"+x for x in sorted(os.listdir(cwd+"/data/"))]
img_path=[datapath+p for p in filename]
print(img_path)

# Batch read images
imgs_gray=read_batch(img_path)
imgs_RGB=read_batch(img_path, sty="RGB")
img_size=imgs_gray[0].shape


# Stitch

num_H=3 # number of points to defined Homography matrix
use_refined_feat=False
out_gray=imgs_gray[0]
out_color=imgs_RGB[0]
for im in range(1, len(imgs_gray)):
    print("================================================")
    print("Start stitching image "+str(im)+" and "+str(im+1))
    imgs_pair_gray=[out_gray,imgs_gray[im]]
    imgs_pair_RGB=[out_color,imgs_RGB[im]]
    idx_H=0
    idx_A=1
    kp_list, des_list=feature_detect(imgs_pair_gray, num_feature=0, contrast_th=0.06)
    print("Keypoints number in each image",len(kp_list[0]),len(kp_list[1]))
    kp_list_new, des_list_new=refine_feature(kp_list,des_list, img_size,3)
    print("Refined keypoints number in each image",len(kp_list_new[0]),len(kp_list_new[1]))
    if use_refined_feat:
        kp_img_H=kp_list_new[idx_H] 
        kp_img_A=kp_list_new[idx_A] 
        des_img_H=des_list_new[idx_H]
        des_img_A=des_list_new[idx_A]
    else:
        kp_img_H=kp_list[idx_H] # image to be H transformed
        kp_img_A=kp_list[idx_A] # anchor image
        des_img_H=des_list[idx_H]
        des_img_A=des_list[idx_A]
    inliner_index, H=ransac(kp_img_H,kp_img_A,
                             des_img_H,des_img_A,
                             itr=2000,th_num=5,tol_pixel=1,num_sample=num_H)
    print("Transformation matrix: \n",H)
    
    coor=cal_homo_coor(imgs_pair_gray[0].shape)
    out_RGB=[]
    out_noH_RGB=[]
    label=["R","G","B"]
    for i in range(3): # loop RGB channels
        print("Render chanel: ",label[i])
        patch, patch_noH=image_transform(H,coor, imgs_pair_RGB[idx_H][:,:,i], imgs_pair_RGB[idx_A][:,:,i])
        out_RGB.append(patch)
        out_noH_RGB.append(patch_noH)
    out_RGB=np.moveaxis(np.array(out_RGB), 0, -1)
    out_noH_RGB=np.moveaxis(np.array(out_noH_RGB), 0, -1)
    # Write image
    cv2.imwrite('out.jpg', out_RGB)
    cv2.imwrite('out_noH.jpg', out_noH_RGB)
#     print(out_RGB.shape, out_noH_RGB.shape)
    out_blend=blend_image(out_RGB,out_noH_RGB,0.7)
    cv2.imwrite('out_blend_{}.jpg'.format(im), out_blend)
    out_gray=read_image('out_blend_{}.jpg'.format(im), style="Gray") # update the gray image
    out_color=out_blend
    print("Finish writing image.")
cv2.imwrite('panorama.jpg', out_blend)

