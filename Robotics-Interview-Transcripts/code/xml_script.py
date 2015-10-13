from os import listdir
from os.path import basename, isfile, join, splitext

raw_dir_path = "C:/trans/txt/"
xml_dir_path = "C:/trans/xml/"


def file_name(path):
    return splitext(basename(path))[0]

for raw_dir_files in [f for f in listdir(raw_dir_path) if isfile(join(raw_dir_path, f))]:
    with open(raw_dir_path+raw_dir_files, "r") as raw_file:
        with open(xml_dir_path + file_name(raw_file.name) + ".xml", "w") as xml:
            xml.write("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n")
            xml.write("<subject>\n<name>" + file_name(raw_file.name) + "</name>\n")
            xml.write("<interview>\n")
            for line in raw_file:
                it=0
                token = line.split(":**")
                if "Interviewer" == token[0]:
                    xml.write("<interviewer>" + token[1].split("\n")[0] + "</interviewer>\n")
                elif 'Interviewee' == token[0]:
                    xml.write("<interviewee>" + token[1].split("\n")[0] + "</interviewee>\n")
                else:
                    print line
                    print token[0] + " " + token[1]
                it += 1
            xml.write("</interview>\n")
            xml.write("</subject>\n")
