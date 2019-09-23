import os
from xml.etree import ElementTree as ET
import pprint as pp


def process_xml_to_dict(xml_filepath):
    data = {}
    xml = ET.parse(xml_filepath)
    root_element = xml.getroot()
    videoName = root_element.find("VideoName").text
    videoName = videoName.replace('\\', '/')
    videoName = videoName.replace("//motionmetrics.net/nas/",
                                  "/media/sf_N_DRIVE/")
    data["video_path"] = videoName
    data["activities"] = {}

    tagCollection = root_element.find("TagCollection")
    tagDisplayers = tagCollection.findall("TagDisplayerSerialize")
    for tagDisplayer in tagDisplayers:
        label = tagDisplayer.find("Description").text
        timeRangeCollection = tagDisplayer.find("TimeRangeCollection")
        timeRangeSerialize = timeRangeCollection.findall("TimeRangeSerialize")
        data["activities"][label] = []
        for t in timeRangeSerialize:
            bframe = int(t.find("BFrame").text)
            eframe = int(t.find("EFrame").text)
            data["activities"][label].append((bframe, eframe))

    return data


if __name__ == "__main__":
    path_to_xml_files = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Farhad/timestamp_labels/Activity/"
    for xml_filename in os.listdir(path_to_xml_files):
        xml_filepath = os.path.join(path_to_xml_files, xml_filename)
        data = process_xml_to_dict(xml_filepath)
        pp.pprint(data)
