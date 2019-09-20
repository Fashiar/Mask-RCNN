import numpy as np
from matplotlib import path
from scipy.ndimage import gaussian_filter
from PIL import Image
import math
import os
import shutil
from tqdm import tqdm
from utils import get_coord, get_random_points, get_bezier_curve, rtnorm
from mycoco_json_utils import create_info, create_license, create_image_key, create_categories, get_segmentation, get_mask_annotation

class fillers_synthesis(object):
    #constructor
    def __init__(self, image_id, coord, img, Nx, Ny):
        self.sub_mask_details = dict()
        self.sub_mask_details['category_id'] = []
        self.sub_mask_details['pixl_arr'] = []

        self.image_id = image_id
        self.coord = coord
        self.img = img
        self.Nx = Nx
        self.Ny = Ny

    def fiber(self, nfiber, center_arr):
        alpha = np.zeros((nfiber, 1))
        L = np.zeros((nfiber, 1))
        W = np.zeros((nfiber, 1))
        Density = np.zeros((nfiber, 1))

        all_fiber_mask = np.zeros((self.Ny, self.Nx))

        for i in range(nfiber):
            alpha[i] = np.random.uniform(-math.pi/2, math.pi/2)
            L[i] = np.random.normal(40, 5) # fiber length mu=40, sd = 5
            W[i] = 4 # constant fiber width 4
            Density[i] = rtnorm(0,1,0.92,0.3)

        for i in range(nfiber):
            #four vertices of a single fiber
            p1 = np.vstack((-W[i]/2,-L[i]/2))
            p2 = np.vstack((W[i]/2,-L[i]/2))
            p3 = np.vstack((W[i]/2,L[i]/2))
            p4 = np.vstack((-W[i]/2,L[i]/2))

            theta = alpha[i]

            R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            center = np.expand_dims(center_arr[i], axis=1)

            # rotation of vertices and move to center
            q1 = np.dot(R,p1) + center
            q2 = np.dot(R,p2) + center
            q3 = np.dot(R,p3) + center
            q4 = np.dot(R,p4) + center

            V_xy = np.hstack((q1,q2,q3,q4))
            p = path.Path(np.transpose(V_xy))
            fiber_pixl = p.contains_points(self.coord)
            fiber_pixl = np.reshape(fiber_pixl, (self.Ny, self.Nx)) - 0.0
            fiber = fiber_pixl*Density[i]

            self.img = np.maximum(self.img, fiber)
            all_fiber_mask = np.maximum(all_fiber_mask, fiber_pixl) # mask of all fibers in a single image

            self.sub_mask_details['category_id'].append(1) # 1 is for fiber category
            self.sub_mask_details['pixl_arr'].append(fiber_pixl)

        return self.img, all_fiber_mask # at this point self.img contains all the fibers per image

    def particles(self, npart, center_arr, rad=0.3, edgy=0):
        all_part_mask = np.zeros((self.Ny, self.Nx))
        for j in range(npart):
            center = center_arr[j]
            a = get_random_points(n=4, scale=30) + center
            s, c = get_bezier_curve(a, rad=rad, edgy=edgy)
            Density = rtnorm(0,1,0.92,0.3)
            
            p = path.Path(c)
            particle_pixl = p.contains_points(self.coord)
            particle_pixl = np.reshape(particle_pixl, (self.Ny, self.Nx)) - 0.0
            particle = particle_pixl*Density
             
            self.img = np.maximum(self.img, particle)
            all_part_mask = np.maximum(all_part_mask, particle_pixl)
            
            self.sub_mask_details['category_id'].append(2) # 2 is for particle category
            self.sub_mask_details['pixl_arr'].append(particle_pixl)
            
        return self.img, all_part_mask, self.sub_mask_details # at this point self.img contains all fibers & particles per image

def validate_args(args):
    assert args.count > 0, 'count must be greater than 0'
    assert args.Image_width >= 64, 'width must be greater than 64 and equal to height'
    assert args.Image_height >= 64, 'height must be greater than 64 and equal to width'
    assert args.Image_width == args.Image_height, 'image width and hight must be same'
    assert args.ncomp > 0, 'ncomp must be greater than 0'
    assert args.blend >= 0 and args.blend <= 1, 'blend must be within 0 to 1'

def process_directories(root):
    directory = os.getcwd() + root
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
    image_path = directory + "images/"
    mask_path = directory + "masks/"
    os.mkdir(image_path)
    os.mkdir(mask_path)
    return image_path, mask_path


def generate_image_and_mask(Ny, Nx, num, image_id, coord, nfiber, npart, sigma):
    image = np.random.uniform(low=0, high=0.4, size=(Ny,Nx))
    C = np.random.randint(10, Nx-20, size=(num,2))
    
    fillers = fillers_synthesis(image_id, coord, image, Nx, Ny)
    fiber_img, fiber_mask = fillers.fiber(nfiber, C[:nfiber])
    part_img, part_mask, sub_mask_details = fillers.particles(npart, C[-npart:])
    
    image = part_img
    mask = np.maximum(fiber_mask, part_mask)
    
    mask = gaussian_filter(mask, sigma)
    image = gaussian_filter(image, sigma)

    return image, mask, sub_mask_details
    
def save_images_and_mask(image, mask, image_id, image_path, mask_path):
    mask_rescaled = (255.0 / mask.max() * (mask - mask.min())).astype(np.uint8)
    image_rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint8)

    mask = Image.fromarray(mask_rescaled)
    mask.save(mask_path + str(image_id+1).zfill(5) +".png")
    image = Image.fromarray(image_rescaled)
    image.save(image_path + str(image_id+1).zfill(5) +".png")

def synthesize_image(args):
    Nx = args.Image_width
    Ny = args.Image_height
    hm_img = args.count
    num = args.ncomp
    mix = args.blend
    sigma = 0.7
    
    nfiber = np.int(mix*num)
    npart = num - nfiber
    coord = np.transpose(get_coord(Nx, Ny))
    
    root = args.output_dir
    image_path, mask_path = process_directories(root)

    should_continue = input('\nwould you like to create dataset annotation json? (y/n): ').lower()

    if should_continue != 'y' and should_continue != 'yes':
        print(f'\nGenerating {hm_img} images and corresponding mask without annotation file...\n')

        for image_id in tqdm(range(hm_img)): 
            #generate individual image and mask     
            image, mask, _ = generate_image_and_mask(Ny, Nx, num, image_id, coord, nfiber, npart, sigma)
            # save each image and mask to root folder
            save_images_and_mask(image, mask, image_id, image_path, mask_path)

        print(f'\nsuccessfully generated {hm_img} images and their masks to folder: \n{image_path} and \n{mask_path}')

    else:
        print(f'\nGenerating {hm_img} images and corresponding mask with annotation file...\n')

        # empty list for .json annotation file
        annotations = []
        image_key = []
        licenses = []

        for image_id in tqdm(range(hm_img)):
            # enerate individual image and mask     
            image, mask, sub_mask_details = generate_image_and_mask(Ny, Nx, num, image_id, coord, nfiber, npart, sigma)
            # save each image and mask to root folder
            save_images_and_mask(image, mask, image_id, image_path, mask_path)

            # each sub mask details are added to the "annotations" list
            for idx in range(len(sub_mask_details['category_id'])):
                sub_mask_id = (idx+1) + image_id*len(sub_mask_details['category_id'])
                segmentation, bbox, area = get_segmentation(sub_mask_details['pixl_arr'][idx])
                annotation = get_mask_annotation(segmentation, 0, image_id+1, sub_mask_details['category_id'][idx], 
                    sub_mask_id, bbox, area)
                annotations.append(annotation)

            image_key.append(create_image_key(image_id+1, Ny, Nx))

        info = create_info() # create the info key
        licenses.append(create_license()) # create the license key
        categories = create_categories(2) # two (2) types of fillers (fiber and particles)

        print(f'\nsuccessfully generated {hm_img} images and their masks to folder: \n{image_path} and \n{mask_path}\n')

        # merge all the key to a single annotation file
        master_file = create_master_file(info, licenses, image_key, annotations, categories)
        # save the annotation file to the "root" directory
        write_master_file(master_file, root)

def create_master_file(info, licenses, image_key, annotations, categories):
    master_file = {
            'info': info,
            'licenses': licenses,
            'images': image_key,
            'annotations': annotations,
            'categories': categories
            }
    return master_file

def write_master_file(master_file, root):
    import json
    output_path = os.getcwd() + root + 'mycoco_instances.json'
    with open(output_path, 'w+') as output_file:
        output_file.write(json.dumps(master_file, indent=4))

    print(f'Annotation successfully written to file:\n{output_path}')

def main(args):
    validate_args(args)
    synthesize_image(args)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Artificial SEM Images")
    parser.add_argument("--Image_width", type=int, required=True, help="Width of the image")
    parser.add_argument("--Image_height", type=int, required=True, help="height of the image")
    parser.add_argument("--count", type=int, required=True, help="Number of images to be generated")
    parser.add_argument("--ncomp", type=int, required=True, help="Number of particles/fiber in a image")
    parser.add_argument("--blend", type=float, required=True, help="percentage of fiber (0 to 1)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save all the images (e.g. '/dataset/')")
    
    args = parser.parse_args()
    
    main(args)