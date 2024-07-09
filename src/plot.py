import numpy as np
import glob
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_with_colorbar(img,vmax=0):
    """
    args:
        vmax: The maximal value to be plotted
    """
    ax = plt.gca()
    if(vmax == 0):
        im = ax.imshow(img, cmap= 'gray')
    else:
        im = ax.imshow(img, cmap= 'gray',vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    

def plot_input_histogram(imgs,sensitivity):
    """
    
    The imgs variable consists of 1 image captured per different camera sensitivity (ISO) settings. plot_input_histogram
    visualize the histograms for each image in a subplot fashion

    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
    
    """
    num_sensitivity = imgs.shape[2]
    
    fig, axes = plt.subplots(1,  num_sensitivity, figsize=(30, 5))
    
    for i in range(num_sensitivity):
        ax = axes[i]
        ax.hist(imgs[:, :, i].ravel(), bins=256, alpha=0.5, range=(0, 255))
        ax.set_title(f'Sensitivity Lvl {sensitivity[i]}')
        ax.set_ylabel('Count')
        ax.grid(True)
    
    axes[-1].set_xlabel('Intensity')
    plt.tight_layout()
        
def plot_histograms_channels(img,sensitivity):
    """
    
    Plots the histogram for each channel in a subplot (1 row, 3 cols)
    
    args:
        img(np.ndarray): The RGB image
        sensitivity(float): The gain settings of the img series
    
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 2.5))
    
    channels = ['Red', 'Green', 'Blue']
    for i, (channel, ax) in enumerate(zip(channels, axes)):
        ax.hist(img[:, :, i].ravel(), bins=50, alpha=0.5, range=(0, 255))
        ax.set_title(f'{channel} Channel')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Count')
        ax.grid(True)
    
    fig.suptitle(f'Histograms for Sensitivity lvl = {sensitivity}')
    plt.tight_layout()

        
def plot_input_images(imgs,sensitivity):
    """
    
    The dataset consists of 1 image captured per different camera sensitivity (ISO) settings. Lets visualize a single image taken at each different sensitivity setting
    
    Hint: Use plot_with_colorbar. Use the vmax argument to have a scale to 255
    (if you don't use the vmax argument)
    
    args:
        imgs(np.ndarray): 3-dimensional array containing one image per intensity setting (not all the 200)
        sensitivity(np.ndarray): The sensitivy (gain) vector for the image database
    
    """
    num_sensitivity = imgs.shape[2]
    fig, axes = plt.subplots(1, num_sensitivity, figsize=(20, 5))
    
    for i in range(num_sensitivity):
        plt.sca(axes[i])
        plot_with_colorbar(imgs[:, :, i], vmax=255)
        axes[i].set_title(f'Sensitivity: {sensitivity[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()




def plot_rgb_channel(img, sensitivity):
    fig, axes = plt.subplots(1, 3, figsize=(20, 11))
    
    vmax_r = img[:, :, 0].max()
    vmax_g = img[:, :, 1].max()
    vmax_b = img[:, :, 2].max()
    
    plt.sca(axes[0])
    plot_with_colorbar(img[:, :, 0], vmax=vmax_r)
    axes[0].set_title('Red Channel')
    
    plt.sca(axes[1])
    plot_with_colorbar(img[:, :, 1], vmax=vmax_g)
    axes[1].set_title('Green Channel')
    
    plt.sca(axes[2])
    plot_with_colorbar(img[:, :, 2], vmax=vmax_b)
    axes[2].set_title('Blue Channel')
    
    fig.suptitle(f'Sensitivity: {sensitivity}', y = .75)
    plt.tight_layout()

def plot_images(data, sensitivity, statistic,color_channel):
    """
    this function should plot all 3 filters of your data, given a
    statistic (either mean or variance in this case!)

    args:

        data(np.ndarray): this should be the images, which are already
        filtered into a numpy array.

        statsistic(str): a string of either mean or variance (used for
        titling your graph mostly.)

    returns:

        void, but show the plots!

    """
    num_sensitivities = data.shape[3]
    
    plt.figure(figsize=(20, 10))
    for i in range(num_sensitivities):
        plt.subplot(2, 3, i+1)
        plt.title(f'Sensitivity {sensitivity[i]} - {statistic}')
        plot_with_colorbar(data[:, :, color_channel, i])

    
    plt.suptitle(f'Images - Color Channel {["red", "green", "blue"][color_channel]}', y = .9)
    # plt.tight_layout()
    plt.show()
    
    
def plot_relations(means, variances, skip_pixel, sensitivity, color_idx):
    """
    this function plots the relationship between means and variance. 
    Because this data is so large, it is recommended that you skip
    some pixels to help see the pixels.

    args:
        means: contains the mean values with shape (200x300x3x6)
        variances: variance of the images (200x300x3x6)
        skip_pixel: amount of pixel skipped for visualization
        sensitivity: sensitivity array with 1x6
        color_idx: the color index (0 for red, 1 green, 2 for blue)

    returns:
        void, but show plots!
    """
    plt.figure(figsize=(20, 5))
    
    for i, sens in enumerate(sensitivity):
        plt.subplot(1, 6, i+1)
        plt.scatter(means[::skip_pixel, ::skip_pixel, color_idx, i].ravel(),
                    variances[::skip_pixel, ::skip_pixel, color_idx, i].ravel(), alpha=0.5)
        plt.xlabel('Mean Intensity')
        plt.ylabel('Variance')
        plt.title(f'Sensitivity {sens}')
        plt.grid(True)
    
    plt.suptitle(f'Mean vs Variance - Color Channel {["red", "green", "blue"][color_idx]}')
    plt.tight_layout()
    plt.show()
        
def plot_mean_variance_with_linear_fit(gain,delta,means,variances,skip_points=50,color_channel=0):
    """
        this function should plot the linear fit of mean vs. variance against a scatter plot of the data used for the fitting 
        
        args:
        gain (np.ndarray): the estimated slopes of the linear fits for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits for each color channel and camera sensitivity

        means (np.ndarray): the means of your data in the form of 
        a numpy array that has the means of each filter.

        variances (np.ndarray): the variances of your data in the form of 
        a numpy array that has the variances of each filter.
        
        skip_points: how many points to skip so the scatter plot isn't too dense
        
        color_channel: which color channel to plot

    returns:
        void, but show plots!
    """
    num_sensitivities = means.shape[3]
    
    plt.figure(figsize=(20, 5))
    
    for i in range(num_sensitivities):
        plt.subplot(1, 6, i+1)
        plt.scatter(means[::skip_points, ::skip_points, color_channel, i].ravel(),
                    variances[::skip_points, ::skip_points, color_channel, i].ravel(), alpha=0.5)
        x_vals = np.linspace(0, np.max(means[:, :, color_channel, i]), 100)
        y_vals = gain[color_channel, i] * x_vals + delta[color_channel, i]
        plt.plot(x_vals, y_vals, color='red')
        plt.xlabel('Mean Intensity')
        plt.ylabel('Variance')
        plt.title(f'Gain = {round(gain[color_channel, i], 3)} | Delta = {round(delta[color_channel, i], 3)}')
        plt.grid(True)
    
    plt.suptitle(f'Mean vs Variance with Linear Fit | Color Channel = {["red", "green", "blue"][color_channel]}')
    plt.tight_layout()
    plt.show()
    
def plot_read_noise_fit(sigma_read, sigma_ADC, gain, delta, color_channel=0):
    """
        this function should plot the linear fit of read noise delta vs. gain plotted against the data used for the fitting 
        
        args:
        sigma_read (np.ndarray): the estimated gain-depdenent read noise for each color channel of the sensor 

        sigma_ADC (np.ndarray): the estimated gain-independent read noise for each color channel of the sensor

        gain (np.ndarray): the estimated slopes of the linear fits of mean vs. variance for each color channel and camera sensitivity

        delta (np.ndarray): the estimated bias/intercept of the linear fits of mean vs. variance for each color channel and camera sensitivity

        color_channel: which color channel to plot
        
    returns:
        void, but show plots!
    """
    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    plt.scatter(gain[color_channel, :], delta[color_channel, :], alpha=0.5, label='Data points')
    
    x_vals = np.linspace(0, np.max(gain[color_channel, :]), 100)
    y_vals = sigma_read[color_channel] * x_vals**2 + sigma_ADC[color_channel]
    plt.plot(x_vals, y_vals, color='red', label='Linear fit')
    
    plt.xlabel('Gain (g)')
    plt.ylabel('Read Noise')
    plt.title(f'Read Noise Fit | Color Channel = {["red", "green", "blue"][color_channel]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()