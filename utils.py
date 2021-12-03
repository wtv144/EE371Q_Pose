import matplotlib.pyplot as plt 
import numpy as np 
import cv2
'''PlotOverlay Using plt (Jupyter friendly)'''
def plotoverlay(data, img,fdir=None):
    adj_mat = {0:[5,6], 1:[0,2,3], 2:[4], 5:[7,11], 6:[8,12], 7:[9], 8:[10], 11:[12,13], 12:[14], 13:[15], 14:[16]} 

    fig,ax = plt.subplots()
    print(fdir)
    for x,y in data:
        ax.add_patch(plt.Circle((x,y), 3,color='r',zorder=5))
    for keypoint in adj_mat:
        for edge in adj_mat[keypoint]:
            plt.plot([data[keypoint][0], data[edge][0]], [data[keypoint][1], data[edge][1]], color='white', linewidth=2)
    plt.imshow(img)
    plt.axis('off')  # command for hiding the axis.

    if fdir != None:
        plt.savefig(fdir,bbox_inches='tight', pad_inches=0)
    plt.close()

'''PlotOverlay using CV2 (not Jupyter Friendly)'''
def plotoverlayCV2(img, data):
    adj_mat = {0:[5,6], 1:[0,2,3], 2:[4], 5:[7,11], 6:[8,12], 7:[9], 8:[10], 11:[12,13], 12:[14], 13:[15], 14:[16]} 
    window_name = 'image'
    
    for keypoint in adj_mat:
        for edge in adj_mat[keypoint]:
            img = cv2.line(img, [data[keypoint][0], data[keypoint][1]], [data[edge][0], data[edge][1]], (255,255,255), 2)
    for x,y in data:
        img = cv2.circle(img, (x,y), 3, (0,0,255), -1)        
    cv2.imshow(window_name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

'''Eval_PCK'''
def eval_PCK(estimate, actual):
    if len(estimate) != len(actual):
        return -1
    pcj_fraction = 0.2
    l_shoulder = actual[5]
    #l_hip = actual[11]
    r_hip = actual[12]
    #torso_height = calc_dist(l_shoulder, l_hip)
    torso_diameter = calc_dist(l_shoulder, r_hip)
    
    #pcj_threshold = pcj_fraction*torso_diameter if torso_diameter>1.05*torso_height else pcj_fraction*bbox_diag 
    pcj_threshold = pcj_fraction*torso_diameter
    
    dists = []
    correct = 0
    for i in range(len(estimate)):
        diff = calc_dist(estimate[i], actual[i])
        dists.append(diff)
        if diff < pcj_threshold:
            correct +=1
    #print(dists)
    return correct/len(estimate)     

'''Distance formula needed for eval_PCK'''
def calc_dist(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5