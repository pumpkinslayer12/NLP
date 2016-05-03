import praw
import pprint
import csv
import re
path1= 'H:/data_files/reddit_urls.txt'
path2='H:/data_files/reddit_dataset.txt'

user_agent='dialect_application by /u/sigma_draconis'
#dialect_detection_social_media_v1
def start():
    with open(path1,'r') as reddit_file:
        with open(path2,'w', encoding='utf-8') as reddit_data_file:
            csv_reader= csv.reader(reddit_file,delimiter=',',quotechar='"')
            r=praw.Reddit(user_agent=user_agent)
            state_buffer=''
            for state, city, sub_url in csv_reader:
                print('Started state: ' +str(state)+ ', city: '+str(city))
                if state_buffer != state:
                    state_buffer= state
                
                subreddit= r.get_subreddit(sub_url).get_top_from_year(limit=10)
                for post in subreddit:
                    #compresses tree as comment order is irrelevant
                    flat_comments= praw.helpers.flatten_tree(post.comments)
                    for comment in flat_comments:
                        if hasattr(comment,'body'):
                            cln_comment=clean_comment(comment.body)
                            reddit_data_file.write('\"'+state+'\",\"'+cln_comment+'\"\n')
                print('Finished state: ' +str(state)+ ', city: '+str(city))
    
def clean_comment(string_comment):
        comment=str.lower(string_comment.replace('\t',' '))
        comment=comment.replace('\n',' ')
        comment=comment.replace('\r',' ')
        comment=comment.replace('\"',' ')
        comment=comment.replace('\'',' ')
        comment=" ".join(comment.split())
        comment=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',comment)
        return comment
        
start()
