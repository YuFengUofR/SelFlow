import glob
import os
import cv2
import imageio
import numpy as np
import torch

FLOW_DIR = "images/kitti_images"
IMAGE_DIR = "images/kitti/image_02/data"

def compensate(rgb_np, optical_flow_np):
	new_img = np.zeros_like(rgb_np)

	shape = optical_flow_np.shape

	print(shape)
	for r in range(shape[0]):
		for c in range(shape[1]):
			new_r = int(r + optical_flow_np[r, c, 1])
			new_c = int(c + optical_flow_np[r, c, 2])
			if new_c < shape[1] and new_c >= 0 and new_r < shape[0] and new_r >= 0:
				new_img[new_r, new_c] = rgb_np[r, c]

	# cv2.imwrite("test.png", new_img)
	return new_img

def read_flo(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=int(2*w*h))
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h[0], w[0],2))
            return data2D   

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    x = torch.unsqueeze(torch.tensor(x, dtype=torch.float64), 0)
    flo = torch.unsqueeze(torch.tensor(flo, dtype=torch.float64), 0)
    x = torch.transpose(x, 1, 3)
    flo = torch.transpose(flo, 2, 3)
    print(x.shape, flo.shape)

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) - flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = torch.nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size(), dtype=torch.float64))
    if x.is_cuda:
        mask = mask.cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    # print(output.shape)
    new_img = output*mask
    new_img = torch.transpose(new_img, 1, 3)
    new_img = torch.squeeze(new_img, 0)
    return new_img

for i in range(1, 11):
    image_file = "%s/%010d.png" % (IMAGE_DIR, i)
    next_file = "%s/%010d.png" % (IMAGE_DIR, i+1)
    flow_file = "%s/flow_fw_%04d.flo" % (FLOW_DIR, i)
    optical_flow_np = read_flo(flow_file)
    optical_flow_np = np.array([optical_flow_np[:, :, 1], optical_flow_np[:, :, 0]])

    rgb_np = cv2.imread(image_file)
    print(image_file)
    # print(rgb_np.shape)
    print(optical_flow_np.shape)

    next_rgb = cv2.imread(next_file)

    # print(optical_flow_np) 
    # new_img = compensate(rgb_np, optical_flow_np)
    new_img = warp(next_rgb, optical_flow_np)

    diff = np.abs(new_img - next_rgb).numpy()

    combine = np.vstack((new_img, next_rgb))
    filename = "%s/comb_%04d.png" % (FLOW_DIR, i)
    cv2.imwrite(filename, combine)

    print(np.mean(diff))


