import argparse
import json
import os.path as osp

# For a given file, will return the supporters found in the posts contained
## Filename is the name of the file that will be read line by line for post content
## Supporter set is the set of supporters that the supporters found will be added to
def get_supporters_from_file(filename, supporter_set):
    filepath = osp.join('596_data/', filename)
    myfile = open(filepath, 'r')
    lines = myfile.readlines()

    for line in lines:
        this_dict = json.loads(line)
        # Verify that there is an author associated with the post
        if 'author' in this_dict:
            author = this_dict['author']
            supporter_set.add(author)
    return supporter_set

# Aggregates all the sets of supporters together for the candidate
## Candidate files are the files that contain the posts from a candidate's subreddit
## Supporter set is the set of supporters of a given candidate to which all the supporters found will be added to
def add_to_supporter_set(candidate_files, supporter_set):
    for candidate_file in candidate_files:
        print(candidate_file)
        supporters = get_supporters_from_file(candidate_file, supporter_set)
        supporter_set = supporter_set.union(supporters)
    return supporter_set

def write_supporters_to_txt(filename, supporter_list):
    with open(filename, 'w') as f:
        for supporter in supporter_list:
            f.write(supporter)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('input_file', help='The path to the JSON file to be parsed')
    #args = parser.parse_args()
    #input_file = args.input_file
    biden_files = ['20201101-biden.json', '20201102-biden.json', '20201103-biden.json', '20201104-biden.json', '20201105-biden.json', '20201106-biden.json']
    trump_files = ['20201101-trump.json', '20201102-trump.json', '20201103-trump.json', '20201104-trump.json', '20201105-trump.json', '20201106-trump.json']

    trump_supporters = set()
    biden_supporters = set()


    biden_supporters = add_to_supporter_set(biden_files, biden_supporters)
    trump_supporters = add_to_supporter_set(trump_files, trump_supporters)
    print(len(biden_supporters))
    print(len(trump_supporters))

    # Make sure we remove any authors that we found in both
    duplicate_supporters = set()
    duplicate_supporters = trump_supporters & biden_supporters
    trump_supporters = trump_supporters - duplicate_supporters
    biden_supporters = biden_supporters - duplicate_supporters

    print(len(biden_supporters))
    print(len(trump_supporters))
    write_supporters_to_txt('biden_supporters.txt', biden_supporters)
    write_supporters_to_txt('trump_supporters.txt', trump_supporters)

if __name__ == '__main__':
    main()
