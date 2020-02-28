import csv
import os
import shutil
#returns tuple of content and header. By default assumes that there is not a header.
def process_csv(csv_path, has_header=0):
    with open(csv_path, 'r', encoding="utf-8") as csv_file:
        csv_reader=csv.reader(csv_file)
        header=[]
        content=[]
        if has_header:
            header=next(csv_reader)
        for row in csv_reader:
            content.append(row)
        return (content, header)

#assumes list in list with first list acting as container and secondary lists as rows of data.
def organize_by_category(raw_list, category_position):
    category_dictionary={}

    for row in raw_list:
        if row[category_position] in category_dictionary:
            category_dictionary[row[category_position]].append(row)
        else:
            category_dictionary[row[category_position]]=[]
            category_dictionary[row[category_position]].append(row)

    return category_dictionary

def create_csv_file_string(raw_list):

 return "\n".join([",".join(row) for row in raw_list])

#Applies proper captilization to all rows and then lower cases the column with email addresses
def string_to_proper(raw_list,email_col):

    return_list=[]
    for row in raw_list:
        title_row=[col.title() for col in row]
        title_row[email_col]=title_row[email_col].lower()
        return_list.append(title_row)
    return return_list

def main():
    csv_file_path="rosters.csv"
    content, header= process_csv(csv_file_path,1)
    print(len(content))
    print(len(header))
    content_by_category=organize_by_category(content,4)

    #processing of content to fix capitlization
    content_by_category={category:string_to_proper(content,2) for category,content in content_by_category.items()}

    content_by_category={category:create_csv_file_string(content) for category,content in content_by_category.items()}

    dir="staff_listing"

    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    for category, content in content_by_category.items():
        with open(dir+"//"+category+".csv", "w+", encoding="utf-8-sig") as csv_file:
            csv_file.write(",".join(header)+"\n")
            csv_file.write(content)
main()

