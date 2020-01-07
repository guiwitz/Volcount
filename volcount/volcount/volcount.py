"""
This module implements a Python class for the interactive analysis
of microscopy images of synapases imaged by fluorescence microscopy.
"""
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3 License


from IPython.display import display, clear_output
from notebook.notebookapp import list_running_servers

import ipywidgets as ipw
from ipywidgets import ColorPicker, VBox, HBox, jslink

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.filters
import skimage.morphology
import skimage.io
import pandas as pd
import scipy.ndimage as ndi
import glob, os
import subprocess

#import napari
import ipyvolume as ipv
from .folders import Folders
    
class Improc:
    
    def __init__(self, folder_name = 'upload_data'):

        """Standard __init__ method.
        
        Parameters
        ----------
        file = str
            file name
        
        
        Attributes
        ----------
            
        files = list
            list of files to process
        
        ax:  AxesSubplot object
        implot : AxesImage object
        
        file: upload widget
        select_file : selection widget
        select_file_to_plot : selection widget
        sigma_slide : float slider widget
        out : output widget
        out_zip : output widget
        process_button : button widget
        zip_button : button widget
        myHTML : HTML widget
        
        """
        
        #self.folder_name = folder_name
        #self.folder_init()
        
        self.ax = None
        self.implot = None
        self.im_region_mask = None
        self.im_objects = None
        self.density = []
        
        
        #create widgets
        #style = {'description_width': '40%','readout_width' : '60%'}
        style = {'description_width': 'initial'}
        layout = {'width': '300px'}
        
        self.folders = Folders(rows = 15)
        
        self.process_button = ipw.Button(description = 'Click to process', style = style, layout = layout)
        self.process_button.on_click(self.do_processing)
        
        self.zip_button = ipw.Button(description = 'Click to zip data')
        self.zip_button.on_click(self.do_zipping)
        
        self.plot_button = ipw.Button(description = 'Click to plot', style = style, layout = layout)
        self.plot_button.on_click(self.on_select_to_plot)
        
        self.scalingfactor = ipw.Text('1', description = 'z scaling')
        
        self.update_button = ipw.Button(description = 'Update visualization',style = style, layout = layout)
        self.update_button.on_click(self.on_select_to_plot)
        
        self.out = ipw.Output()
        with self.out:
            display(ipv.figure())        
        
        my_adress = next(list_running_servers())['base_url']
        self.myHTML = ipw.HTML("""<a href="https://hub.gke.mybinder.org"""+my_adress+"""notebooks/to_download.tar.gz" target="_blank"><b>5. Hit this link to download your data<b></a>""")

     
    def on_scale_change(self,b):
        
        self.volume_image.extent[2] = (-1024/(int(self.scalingfactor.value)),1024/(int(self.scalingfactor.value)))
        
    def on_select_to_plot(self, b):
        """Call-back function for plotting a 3D visualisaiton of the segmentation"""
        
        self.out.clear_output()
        
        image = skimage.io.imread(self.folders.cur_dir.as_posix()+'/'+self.folders.file_list.value[0], plugin = 'tifffile')
        image2 = skimage.io.imread(self.folders.cur_dir.as_posix()+'/'+os.path.splitext(self.folders.file_list.value[0])[0]+'_label.tif', plugin = 'tifffile')
        image3 = skimage.io.imread(self.folders.cur_dir.as_posix()+'/'+os.path.splitext(self.folders.file_list.value[0])[0]+'_region.tif', plugin = 'tifffile')
        
        scalez = 1024/(int(self.scalingfactor.value))
        xy_extent = [0,1024]
        #create ipyvolume figure
        with self.out:
            ipv.figure()
            self.volume_image = ipv.volshow(image[0,:,:,:,1],extent=[xy_extent,xy_extent,[-scalez,scalez]],level=[0.3, 0.2,0.2], 
                           opacity = [0.2,0,0],controls=False)
            self.volume_seg = ipv.plot_isosurface(np.swapaxes(image2>0,0,2),level=0.5,controls=False, color='green',extent=[xy_extent,xy_extent,[-scalez,scalez]])
            self.volume_reg = ipv.plot_isosurface(np.swapaxes(image3>0,0,2),level=0.5,controls=False, color='blue', extent=[xy_extent,xy_extent,[-scalez,scalez]])
            
            self.volume_reg.brightness = 10
            self.volume_image.brightness = 10
            self.volume_image.opacity = 100 
            ipv.xyzlim(0,1024)
            ipv.zlim(-500,500)
            ipv.style.background_color('white')
            
            minval_data = ipw.IntSlider(min=0, max=255,value = 255, description = 'min val')
            maxval_data = ipw.IntSlider(min=0, max=255,value = 255, description = 'max val')
            brightness_data = ipw.FloatSlider(min=0, max=100,value = 7.0, description = 'brightness')
            opacity_data = ipw.FloatSlider(min=0, max=100,value = 7.0, description = 'opacity')
            level_data = ipw.FloatSlider(min=0, max=1,value = 0.3,step = 0.01, description = 'level')
            levelwidth_data = ipw.FloatSlider(min=0, max=1,value = 0.1,step = 0.01, description = 'level width')

            color = ColorPicker(description = 'Segmentation color')
            color2 = ColorPicker(description = 'Segmentation color')

            visible_seg = ipw.Checkbox()
            visible_reg = ipw.Checkbox()

            jslink((self.volume_image, 'show_min'), (minval_data, 'value'))
            jslink((self.volume_image, 'show_max'), (maxval_data, 'value'))
            jslink((self.volume_image, 'brightness'), (brightness_data, 'value'))
            jslink((self.volume_image.tf, 'opacity1'), (opacity_data, 'value'))
            jslink((self.volume_image.tf, 'level1'), (level_data, 'value'))
            jslink((self.volume_image.tf, 'width1'), (levelwidth_data, 'value'))
            jslink((self.volume_seg, 'color'), (color, 'value'))
            jslink((self.volume_reg, 'color'), (color2, 'value'))
            jslink((self.volume_seg, 'visible'), (visible_seg, 'value'))
            jslink((self.volume_reg, 'visible'), (visible_reg, 'value'))
            ipv.show()
            
            image_controls = HBox([VBox([minval_data, maxval_data]),VBox([brightness_data,opacity_data,level_data,levelwidth_data])])
            display(VBox([HBox([color,visible_seg]),HBox([color2,visible_reg]), image_controls]))
            
        
            #viewer = napari.Viewer(ndisplay = 3)
            #viewer.add_image(image, colormap = 'red')
            #viewer.add_image(image2, colormap = 'green', blending = 'additive')
            #viewer.add_image(image3, colormap = 'blue', blending = 'additive')


       
    def do_processing(self, b):
        """Call-back function for proessing button. Executes image processing and saves result."""

        self.process_button.description = 'Processing...'
        for f in self.folders.file_list.value:
            if f != 'None':
                image = self.import_image(self.folders.cur_dir.as_posix() + "/" + f)
                synapse_region = self.find_synapse_area(image)
                synapse_mask = self.detect_synapses(image, synapse_region)
                synapse_label = skimage.morphology.label(synapse_mask)

                #measure regions
                synapse_regions = pd.DataFrame(skimage.measure.regionprops_table(synapse_label,properties=('label','area')))
                #calculate density as ration of # of synapses per pixels in the synapse region mask
                density = np.sum(synapse_regions.area>50)/np.sum(synapse_mask)

                self.density.append({'filename': os.path.splitext(f)[0], 'density': density*100})

                skimage.io.imsave(self.folders.cur_dir.as_posix()+'/'+os.path.splitext(f)[0]+'_label.tif', synapse_label.astype(np.uint16))
                skimage.io.imsave(self.folders.cur_dir.as_posix()+'/'+os.path.splitext(f)[0]+'_region.tif', synapse_region.astype(np.uint16))
        
        pd.DataFrame(self.density).to_csv(self.folders.cur_dir.as_posix()+'/summary.csv')
        
        self.process_button.description = 'Done! Click to process again'
        
        
    def do_zipping(self, b):
        """zip the output"""
        
        self.zip_button.description = 'Currently zipping...'
        #save the summary file
        pd.DataFrame(self.density).to_csv(self.folders.cur_dir.as_posix()+'/summary.csv')
        
        subprocess.call(['tar', '-czf', 'to_download.tar.gz','-C', self.folders.cur_dir.as_posix(),'.'])
        subprocess.call(['tar', '-czf', 'to_download.tar.gz','-C', self.folders.cur_dir.parent.as_posix(), self.folders.cur_dir.name])
        self.zip_button.description = 'Finished zipping!'
        

    def import_image(self, file):
        """Load file
        
        Parameters
        ----------
        file : str
            name of file to open
        Returns
        -------
        image: 3D numpy array
        
        """

        image = skimage.io.imread(file, plugin='tifffile')
        image = image[0,:,:,:,1]
        return image


    def find_synapse_area(self, image):
        """Finds large scale region where synapases are present
        
        Parameters
        ----------
        image : 3D numpy array

        Returns
        -------
        large_mask_dil: 3D numpy array
            mask of snyapase region
        
        """
        
        #smooth image on large scale to find region of synapses (to remove the dotted structures)
        image_gauss = skimage.filters.gaussian(image, 10)
        #create large mask of synapase region
        large_mask = image_gauss > skimage.filters.threshold_otsu(image_gauss)
        #find the largest region and define it as synapse region
        large_label = skimage.morphology.label(large_mask)
        regions = pd.DataFrame(skimage.measure.regionprops_table(large_label,properties=('label','area')))
        lab_to_keep = regions.sort_values(by = 'area', ascending = False).iloc[0].label
        large_mask2 = large_label == lab_to_keep
        #dilate the region to fill holes
        large_mask_dil = skimage.morphology.binary_dilation(large_mask2, np.ones((5,5,5)))

        return large_mask_dil

    def detect_synapses(self, image, synapse_region):
        """Finds large scale region where synapases are present
        
        Parameters
        ----------
        image : 3D numpy array
        synapse_region : 3D numpy array
            mask of synapse region

        Returns
        -------
        synapse_mask: 3D numpy array
            mask of synapses
        
        """
        
        #calculate a LoG image to highlight synapses
        image_log = ndi.gaussian_laplace(-image.astype(float), sigma=1)
        #within the synapse region calculate a threshold
        newth = skimage.filters.threshold_otsu(image_log[synapse_region])
        #create a synapse mask (and use the global mask too)
        synapse_mask = synapse_region*(image_log > newth)

        return synapse_mask
    
    
     

        
        
        
        
        
            
            