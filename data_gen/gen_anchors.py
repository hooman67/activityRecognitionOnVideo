import random
import argparse
import numpy as np

from preprocessing import parse_annotation
import json

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-a',
    '--anchors',
    default=9,
    type=int,
    help='number of anchors to use')

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.2f,%0.2f, ' % (anchors[i,0], anchors[i,1])

    #there should not be comma after last anchor, that's why
    r += '%0.2f,%0.2f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"

    print(r)

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    hsCount = 0
    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        absSumOfDistances = np.sum(np.abs(old_distances-distances))
        print("iteration {}: dists = {}".format(iteration, absSumOfDistances))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def main(argv):
    config_path = args.conf
    initial_num_anchors = args.anchors
    print("Generating %d anchors..." % initial_num_anchors)

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())


    print('labels:\n')
    print(config['model']['labels'])

    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    grid_w = config['model']['input_size']/32
    grid_h = config['model']['input_size']/32


    


    #anchorsFinalAcceptableIou = 0.9
    #anchorsAvgIou = 0
    #num_anchors = initial_num_anchors
    #hsItCounter = 0

    #while anchorsAvgIou < anchorsFinalAcceptableIou:
        # run k_mean to find the anchors
    annotation_dims = []
    num_wrong = 0
    for image in train_imgs:
        cell_w = image['width']/grid_w
        cell_h = image['height']/grid_h

        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/cell_w
            relative_h = (float(obj["ymax"]) - float(obj['ymin']))/cell_h
            if not (relative_w < 0 or relative_h < 0):
                annotation_dims.append(tuple(map(float, (relative_w,relative_h))))
            else:
                num_wrong += 1
                print ("WRONG! ", num_wrong)

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, initial_num_anchors)

    # write anchors to file
    anchorsAvgIou = avg_IOU(annotation_dims, centroids)
    print('\naverage IOU for', initial_num_anchors, 'anchors:', '%0.2f' % anchorsAvgIou)
    print_anchors(centroids)
        #hsItCounter += 1
        #num_anchors += 1
        #print('\nThis was iteration: ' + str(hsItCounter) + '  NumAnchors is now increased to  ' +  str(num_anchors)+'\n\n')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
