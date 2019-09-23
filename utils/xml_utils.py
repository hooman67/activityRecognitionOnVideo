import xml.etree.cElementTree as ET
import os


def convert_to_pixels(bbox, image_w=640, image_h=640):
    xmin = bbox.xmin * image_w
    xmax = bbox.xmax * image_w
    ymin = bbox.ymin * image_h
    ymax = bbox.ymax * image_h
    return xmin, xmax, ymin, ymax


def soft_clip_coords(xmin, ymin, xmax, ymax):
    if xmin > -18 and xmin < 0: xmin = 0.
    if ymin > -18 and ymin < 0: ymin = 0.
    return int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))


def assert_coords(xmin, ymin, xmax, ymax, label):
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0 or \
            xmin > xmax or ymin > ymax or \
            xmax > 730 or ymax > 490:
        print("Wrong Coordinates of %s: %.1f, %.1f, %.1f, %.1f" % (
        label, xmin, ymin, xmax, ymax))
        return False
    else:
        return True


def gen_xml_file(img_path, bboxes, labels, path_to_write_xml,
                 excluded_classes=["Tooth", "Toothline"]):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "image"
    image_name = img_path.split('/')[-1]
    ET.SubElement(annotation, "filename").text = image_name
    ET.SubElement(annotation, "path").text = img_path
    ET.SubElement(annotation, "video_path").text = "EMPTY"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(640)
    ET.SubElement(size, "height").text = str(480)
    ET.SubElement(size, "depth").text = str(3)
    ET.SubElement(annotation, "segmented").text = str(0)

    bucket_present = False
    for bbox in bboxes:
        label = labels[bbox.get_label()]
        if label == "BucketBB": bucket_present = True

    for i, bbox in enumerate(bboxes):
        if bbox.filtered: continue
        label = labels[bbox.get_label()]
        if label in excluded_classes: continue
        object = ET.SubElement(annotation, "object")
        ET.SubElement(object, "name").text = label
        ET.SubElement(object, "truncated").text = str(0)
        ET.SubElement(object, "difficult").text = str(0)
        bndbox = ET.SubElement(object, "bndbox")
        xmin, xmax, ymin, ymax = convert_to_pixels(bbox, image_w=640, image_h=640)
        ymin *= 480. / 640  # TODO
        ymax *= 480. / 640
        xmin, ymin, xmax, ymax = soft_clip_coords(xmin, ymin, xmax, ymax)
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
        ET.SubElement(bndbox, "p").text = str(bbox.get_score())
        if not assert_coords(xmin, ymin, xmax, ymax, label):
            print("img %s wrong!" % image_name)
            annotation.remove(object)

        if not bucket_present and label == "MatInside":
            print("==========" * 10)
            print("\n=======Creating a bucket!=============")
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = "BucketBB"
            ET.SubElement(object, "truncated").text = str(0)
            ET.SubElement(object, "difficult").text = str(0)
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)
            ET.SubElement(bndbox, "p").text = str(bbox.get_score())
            if not assert_coords(xmin, ymin, xmax, ymax, label):
                print("img %s wrong!" % image_name)
                annotation.remove(object)

    tree = ET.ElementTree(annotation)
    new_xml_filepath = os.path.join(path_to_write_xml, image_name[:-3] + "xml")
    tree.write(new_xml_filepath)

    print("Finished generating XML in VOC format!\n")

    return
