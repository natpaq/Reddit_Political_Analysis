import csv
import json
import os.path as osp
import re
from textblob import TextBlob

# For a given file, will return the supporters found in the posts contained
## Filename is the name of the file that will be read line by line for post content
## Supporter set is the set of supporters that the supporters found will be added to
def get_supporters_from_file(filename, supporter_set, file_writer, candidate):
    filepath = osp.join('596_data/', filename)
    with open(filepath, 'r') as myfile:
        lines = myfile.readlines()
        for line in lines:
            this_dict = json.loads(line)
            # Do a sentiment analysis on the post content
            post_title = this_dict['title']
            post_text = this_dict['selftext']
            analysis = TextBlob(post_title).sentiment
            polarity = analysis[0]
            subjectivity = analysis[1]
            search_string, opponent_string = [], []
            if candidate == 'biden':
                search_string = 'joe|biden'
                opponent_string = 'donald|trump'
            else:
                search_string = 'donald|trump'
                opponent_string = 'joe|biden'
            for post_type in [post_title, post_text]:
            # Only consider positive posts where authors mention the candidate whose sub they posted on negative posts that mentioned the opposing candidate
                if (re.search(search_string, post_type, re.IGNORECASE) and polarity > 0) or (re.search(opponent_string, post_type, re.IGNORECASE) and polarity < 0):
                    # Write these potential posts to a file
                    row = [post_title, polarity, subjectivity, this_dict['author']]
                    file_writer.writerow(row)

            # Verify that there is an author associated with the post
                    if 'author' in this_dict:
                        author = this_dict['author']
                        supporter_set.add(author)
    return supporter_set

# Aggregates all the sets of supporters together for the candidate
## Candidate files are the files that contain the posts from a candidate's subreddit
## Supporter set is the set of supporters of a given candidate to which all the supporters found will be added to
## Candidate --> string specifying candidate we're currently looking at data for
def add_to_supporter_set(candidate_files, supporter_set, candidate):
    for candidate_file in candidate_files:
        #print(candidate_file)
        # Open file to potentially write supporters posts to
        with open(f'user_posts/{candidate_file[:-5]}.csv', 'w', newline="") as f:
            headers = ["Post Title", "Polarity", "Subjectivity", "Author"]
            writer = csv.writer(f)
            writer.writerow(headers)
            supporters = get_supporters_from_file(candidate_file, supporter_set, writer, candidate)
            supporter_set = supporter_set.union(supporters)
    return supporter_set

def write_supporters_to_txt(filename, supporter_list):
    with open(filename, 'w') as f:
        for supporter in supporter_list:
            f.write(supporter)
            f.write('\n')

def main():
    candidates = ['biden', 'trump']

    trump_supporters = set()
    biden_supporters = set()

    for candidate in candidates:
        files = [f'20201101-{candidate}.json', f'20201102-{candidate}.json', f'20201103-{candidate}.json', f'20201104-{candidate}.json', f'20201105-{candidate}.json', f'20201106-{candidate}.json']
        supporters = set()
        supporters = add_to_supporter_set(files, supporters, candidate)
        if candidate == 'biden':
            biden_supporters = supporters
        else:
            trump_supporters = supporters
    #print(len(biden_supporters))
    #print(len(trump_supporters))

    # Make sure we remove any authors that we found in both
    duplicate_supporters = set()
    duplicate_supporters = trump_supporters & biden_supporters
    trump_supporters = trump_supporters - duplicate_supporters
    biden_supporters = biden_supporters - duplicate_supporters

    #print(len(biden_supporters))
    #print(len(trump_supporters))
    write_supporters_to_txt('biden_supporters.txt', biden_supporters)
    write_supporters_to_txt('trump_supporters.txt', trump_supporters)

if __name__ == '__main__':
    main()
