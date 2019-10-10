#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy  as np
import h5py
import os
from sklearn.preprocessing import MinMaxScaler


# In[2]:


path = 'coin.obj.stl.ply'


# In[3]:


# Will process a txt/.ply/.pcd  file containing xyz coords of points and returns a list of
# numpy arrays of each point. Assumes that the input file is the format that
# CloudCompare exports point files as
def process_pt_file(path):
    data = []
    # Process single point cloud file
    pt_file = open(path, 'r')
    # Pass over header lines
    line='placeholder'
    num_pts=0
    if(path.endswith( '.txt')):  
        header = pt_file.readline()
        num_pts = int(pt_file.readline())
        line = pt_file.readline()
    elif (path.endswith( '.ply')):  
        while line:
            line_list = line.split()
            line = pt_file.readline()
            if('element' in line_list[0] and 'vertex' in line_list[1] ):
                num_pts = int(line_list[2])
            if('end_header' in line_list[0]):
                break
    elif (path.endswith( '.pcd')):
        while line:
            line_list = line.split()
            line = pt_file.readline()
            if('POINTS' in line_list[0]   ):
                num_pts = int(line_list[1])
            if('DATA' in line_list[0] and 'ascii' in line_list[1]):
                break  
    previousEleNum = 0
    currentEleNum = 0
   
    while line:
        line_list = line.split()
        data.append(np.array([float(line_list[0]), float(line_list[1]), float(line_list[2])]))
        line = pt_file.readline()
        previousEleNum=currentEleNum
        currentEleNum=len(line.split())  
        if(currentEleNum>6 or (previousEleNum>0 and currentEleNum != previousEleNum)):
            break
       
       
    # Check number of points
    data = np.array(data)
    checkStr="Number of processed points %d   header number of points %d"%(np.shape(data)[0],num_pts)
    print(checkStr)
    #assert num_pts == np.shape(data)[0], checkStr
    return data


# In[4]:


process_pt_file(path)


# In[5]:


def convert_data(path=''):
    if(len(path)<1): #Command line mode
        if len(sys.argv) != 2:
            print("Error: Incorrect number of input arguments. Input only a single argument, either a single pt cloud or a single directory                     containing multiple point clouds")
            exit()
        path = sys.argv[1]
    root, ext = os.path.splitext(os.path.basename(path))    
    npyFilePath = os.path.join(os.path.dirname(path), root + '.npy')
    if os.path.isdir(path):
        print("Processing directory")
        # Process directory of point cloud files
        total_data = None
        for pt_file in os.listdir(path):
            print("Processing file ", pt_file)
            full_path = os.path.join(path, pt_file)
            if not os.path.isfile(full_path):
                print("not file")
                continue
            else:
                curr_data = process_pt_file(full_path)
                print("curr_data shape: ", np.shape(curr_data))
                if total_data is None:
                    total_data = curr_data
                else:
                    total_data = np.vstack((curr_data, total_data))
        np.save(npyFilePath, total_data)
        return total_data
    elif os.path.isfile(path):
        print("Processing single file")
        data = process_pt_file(path)
        np.save(npyFilePath, data)
        return data
    else:
        print("Error: File/directory "+path+" does not exist.")
        exit()


# In[6]:


convert_data(path)


# In[7]:


dataset = convert_data(path)


# In[8]:


XYZVector = dataset
newXYZVector =  XYZVector-np.mean(XYZVector)
maxABScorrdinateValueofAllpoint = np.max(np.absolute(newXYZVector))
newXYZVector = newXYZVector / maxABScorrdinateValueofAllpoint
print(newXYZVector)


# In[9]:


open('newXYZVector.npy', mode='w')
np.save('newXYZVector.npy', newXYZVector)


# In[10]:


with open('newXYZVector.npy','rb') as f:
    arr = np.load(f)
    print(arr)


# In[11]:


# Makes point clouds and saves them to a .h5 file
# Input: 'total_data' np array containing the set of points the point clouds should be subsampled from
# 'num_clouds' is the desired number of point clouds to create
# 'num_pts' is the number of points each point cloud should have. Note that this must be less than or equal to
# the number of points in 'total_data'
# 'filename' name of the .h5 file the data should be saved too
def make_pt_clouds(total_data, num_clouds, num_pts, filename,randomizePoint=True):
   
    f = h5py.File(filename, 'w')
   
    if(randomizePoint):
        rand_inds = np.random.randint(np.shape(total_data)[0], size = num_pts)
    else:
        rand_inds=range(num_pts)
    pt_data = np.reshape(total_data[rand_inds, :], (1, num_pts, 3))
    pt_label = np.array([[40]]*num_clouds)
    for i in range(num_clouds - 1):
        rand_inds = np.random.randint(np.shape(total_data)[0], size = num_pts)
        curr_cloud = np.reshape(total_data[rand_inds, :], (1, num_pts, 3))
        pt_data = np.concatenate((pt_data, curr_cloud), axis=0)
    f.create_dataset('data', data=pt_data)
    f.create_dataset('label', data=pt_label)
    f.close()
    print("pt_data shape: ", np.shape(pt_data))


# In[12]:


total_data = arr


# In[13]:


make_pt_clouds(total_data, 1024, 1024, 'test_coin101.h5',randomizePoint=True)


# In[15]:


f = h5py.File(r'test_coin101.h5', 'r')

# List all groups
print("Keys: %s" % f.keys())
print (list (f))
pclKey = list(f.keys())[0]
#print (pclKey)
labelKey = list(f.keys())[1]
#print (labelKey)
        
# Get the data 
dataNP=np.asarray(f[pclKey])
print (dataNP.shape)
labelNP=np.asarray(f[labelKey])
print (labelNP.shape)


# In[ ]:




