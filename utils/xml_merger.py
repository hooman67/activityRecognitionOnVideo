import os

tooth_xml_folder = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/distilled_labels/hydraulic_bucyrus_p_h/labels"
other_xml_folder = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/distilled_labels/hydraulic_bucyrus_p_h/soft_bucket_labels/"
save_folder = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/distilled_labels/hydraulic_bucyrus_p_h/merged_labels"

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

for i, filename in enumerate(os.listdir(tooth_xml_folder)):
    if i % 100 == 0: print("%d/%d" % (i, len(os.listdir(tooth_xml_folder))))
    tooth_filepath = os.path.join(tooth_xml_folder, filename)
    other_filepath = os.path.join(other_xml_folder, filename)
    with open(tooth_filepath, 'r') as f:
        tooth_xml_text = f.readline()
    with open(other_filepath, 'r') as f:
        other_xml_text = f.readline()

    tooth_xml_text = tooth_xml_text.split("</annotation>")[0]
    other_xml_text = other_xml_text.split("/segmented>")[1]

    full_xml = tooth_xml_text + other_xml_text

    save_filepath = os.path.join(save_folder, filename)
    with open(save_filepath, 'w') as f:
        f.write(full_xml)
