import json
import os.path as osp
import re
import requests

def get_user_data(username, post_type, candidate):
    user_sub_filename = f'{username}_{post_type}.json'
    file_path = osp.join('user_data/' + candidate + '/' + user_sub_filename)
    # Check whether posts for user have already been extracted
    if osp.exists(file_path):
        with open(file_path, 'r') as fread:
            #print("loading ur shit")
            file_posts = fread.readlines()
        return file_posts
    else:
        posts = get_user_posts(username, 100, post_type)
        with open(file_path, "w") as outfile:
            write_to_JSON(posts, outfile)
        return posts

# author_name --> the username of the Reddit user
# num_posts --> the number of entries we want to extract using the API
# sub_or_com --> String value determining whether we're extracting submissions or comments
def get_user_posts(author_name, num_posts, sub_or_com):
    try:
        url = f'https://api.pushshift.io/reddit/search/{sub_or_com}/?author={author_name}&sort=desc&before=12m&size={num_posts}'
        print(url)
        data = requests.get(url, headers={'User-Agent': 'Ubuntu: requests (by /u/Rise_Above_The_Flame)'})

        content = data.json()['data']
        return content

    except:
        print("Error getting data from Pushshift API")
        exit()

# posts --> comments or posts to be added to the dict
# posts_string --> either 'comment' or 'submission' depending on which we're processing
# dict_name --> dict storing user subreddit info
def get_subreddits(posts, posts_string, dict_name):
    for post in posts:
        if type(post) != dict:
            post = json.loads(post)
        subreddit = post['subreddit']
        if subreddit in dict_name:
            if posts_string in dict_name[subreddit]:
                dict_name[subreddit][posts_string] += 1
            else:
                dict_name[subreddit][posts_string] = 1
        else:
            dict_name[subreddit] = dict()
            if posts_string == 'submissions':
                dict_name[subreddit][posts_string] = 1
                dict_name[subreddit]['comments'] = 0
            else:
                dict_name[subreddit][posts_string] = 1
                dict_name[subreddit]['submissions'] = 0
    return dict_name

# Write to JSON file
def write_to_JSON(posts, output_file):
# posts --> posts returned from Reddit api
# output_file --> file object
		# Write each of these posts to output file on a new line
		for post in posts:
			json.dump(post, output_file)
			output_file.write('\n')

# Remove subreddits that don't have much activity on them from a dict
def remove_less_active_subs(users_dict, min_comments, min_submissions):
    to_delete = list()
    for subreddit in users_dict.keys():
        if users_dict[subreddit]['comments'] < min_comments and users_dict[subreddit]['submissions'] < min_submissions:
            to_delete.append(subreddit)
        elif users_dict[subreddit]['comments'] == 0:
            to_delete.append(subreddit)

    # Delete the subreddits that aren't very active for a user
    for subreddit in to_delete:
            users_dict.pop(subreddit, None)

    return users_dict

def remove_political_subs(input_dict):
    political_subreddits = ['JoeBiden',
                            'trump',
                            'donaldtrump',
                            'The_Donald',
                            'Republican',
                            'politics',
                            'Conservative',
                            'ConservativesOnly',
                            'worldpolitics',
                            'conservatives',
                            'askaconservative',
                            'ConservativeMemes',
                            'PoliticalHumor',
                            'PoliticalCompassMemes',
                            'AskALiberal',
                            'LadiesForTrump',
                            'AskTrumpSupporters',
                            'Donald_Trump',
                            'TrumpJR2020',
                            'Trumpvirus',
                            'TrumpCovidFailure',
                            'hottiesfortrump',
                            'TheTrumpZone',
                            'DonaldTrump20',
                            'EnoughTrumpSpam',
                            'VotingForTrump',
                            'real_trumpers',
                            'Trumpgret',
                            'DonaldJTrumpFanClub',
                            'BlackVoicesForTrump',
                            'bidenbro',
                            'BidenRegret',
                            'BidenIsFinished',
                            'JoeBidenSucks',
                            'ShitPoliticsSays',
                            'PoliticalMemes',
                            'ukpolitics',
                            'PoliticalDiscussion',
                            'TexasPolitics',
                            'Political_Tumor',
                            'PoliticalVideo',
                            'Liberal',
                            'liberalgunowners',
                            'democrats',
                            'Kamala',
                            'Pete_Buttigieg',
                            'neoliberal',
                            'LateStageCapitalism',
                            'Impeach_Trump',
                            'TheRightCantMeme',
                            'The_Mueller',
                            'WAlitics',
                            'TheLeftCantMeme',
                            'centrist',
                            'LouderWithCrowder',
                            'libertarianmeme',
                            'Libertarian',
                            'Enough_Sanders_Spam',
                            'AmericanFascism2020',
                            'YangForPresidentHQ',
                            'AskConservatives',
                            'HeckOffCommie',
                            'AskThe_Donald',
                            'Democrats2020',
                            'VoteDEM',
                            'Forum_Democratie',
                            'republicans',
                            'POLYTICAL',
                            'SandersForPresident',
                            ]
    for subreddit in political_subreddits:
        input_dict.pop(subreddit, None) # i dont want to remove this and instead want to annotate it as political
    return input_dict

def main():
    # To store subreddit info by user
    all_users = dict()
    # To aggregate overall subreddit counts
    subreddit_freq = dict()
    subreddit_freq_global = dict()
    candidate = 'trump'
    with open(f'{candidate}_supporters.txt', 'r') as fread:
        users = fread.readlines()
        for user in users:
            user_stats = dict()
            user = user.strip()

            # Part of code used to look for potential bot accounts to be investigated
            # Biden supporter potential bots --> Bakab0t
            #if (re.search('bot|b0t', user, re.IGNORECASE) is not None):
            #    print("BOT ALERT")
            #    print(user)

            # Either scrape or retrieve user data from cache
            user_submissions = get_user_data(user, 'submission', candidate)
            user_comments = get_user_data(user, 'comment', candidate)

            # Get user subreddit count to add to user dictionary
            updated_dict = get_subreddits(user_submissions, 'submissions', user_stats)
            subreddit_frequency = get_subreddits(user_comments, 'comments', updated_dict)

            # Aggregate the total subreddit frequency for all candidate supporters
            new_dict = get_subreddits(user_submissions, 'submissions', subreddit_freq)
            subreddit_freq = get_subreddits(user_comments, 'comments', new_dict)

            # Add user subreddit info to overall user dict
            all_users[user] = updated_dict
            # Disregard subreddits where users have made less than 5 comments and 5 posts
            all_users[user] = remove_less_active_subs(all_users[user], 5, 5)
            # do not remove the political ones just yet!
            # all_users[user] = remove_political_subs(all_users[user])


    # Remove less active subreddits from overall subreddit frequency dict
    subreddit_freq = remove_less_active_subs(subreddit_freq, 50, 50)

    # store the global view just to check
    subreddit_freq_global = subreddit_freq.copy()

    # Remove obviously political subreddits
    subreddit_freq = remove_political_subs(subreddit_freq)

    with open(f'user_data/{candidate}_subreddit_freq.json', 'w') as outfile:
        json.dump(subreddit_freq, outfile, indent=2)

    with open(f'user_data/{candidate}_subreddit_freq_global.json', 'w') as outfile:
        json.dump(subreddit_freq_global, outfile, indent=2)

    with open(f'user_data/{candidate}_user_subred_freq.json', 'w') as outfile:
        json.dump(all_users, outfile, indent=2)


if __name__ == '__main__':
    main()
